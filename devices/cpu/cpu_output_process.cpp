// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_output_process.h"
#include "cpu_output_process_f32_ispc.h"
#include "cpu_output_process_f16_ispc.h"
#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  CPUOutputProcess::CPUOutputProcess(CPUEngine* engine, const OutputProcessDesc& desc)
    : OutputProcess(desc),
      engine(engine)
  {
    if (srcDesc.dataType != DataType::Float32 && srcDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported output process source data type");
  }

  void CPUOutputProcess::submitKernels(const Ref<CancellationToken>& ct)
  {
    check();

    ispc::CPUOutputProcessKernel kernel;

    kernel.src = *src;
    kernel.dst = *dst;
    kernel.tile = toISPC(tile);
    kernel.transferFunc = toISPC(*transferFunc);
    kernel.hdr = hdr;
    kernel.snorm = snorm;

    auto kernelFunc = (src->getDataType() == DataType::Float16)
      ? ispc::CPUOutputProcessKernel_run_f16
      : ispc::CPUOutputProcessKernel_run_f32;

    engine->submitFunc([=]
    {
      parallel_for(kernel.tile.H, [&](int h)
      {
        kernelFunc(&kernel, h);
      });
    }, ct);
  }

OIDN_NAMESPACE_END