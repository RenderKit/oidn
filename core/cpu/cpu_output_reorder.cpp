// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_output_reorder.h"
#include "output_reorder_kernel_ispc.h"

namespace oidn {

  CPUOutputReorderNode::CPUOutputReorderNode(const Ref<CPUDevice>& device, const OutputReorderDesc& desc)
    : CPUNode(device, desc.name),
      OutputReorderNode(desc) {}

  void CPUOutputReorderNode::execute()
  {
    assert(tile.hSrcBegin + tile.H <= src->dims[1]);
    assert(tile.wSrcBegin + tile.W <= src->dims[2]);
    //assert(tile.hDstBegin + tile.H <= output->height);
    //assert(tile.wDstBegin + tile.W <= output->width);

    ispc::OutputReorder kernel;

    kernel.src = *src;
    kernel.output = *output;
    kernel.tile = tile;
    kernel.transferFunc = *transferFunc;
    kernel.hdr = hdr;
    kernel.snorm = snorm;

    parallel_nd(kernel.tile.H, [&](int h)
    {
      ispc::OutputReorder_kernel(&kernel, h);
    });
  }

} // namespace oidn