// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_image_copy.h"
#include "cpu_image_copy_ispc.h"

namespace oidn {

  CPUImageCopy::CPUImageCopy(const Ref<CPUDevice>& device) : CPUOp(device) {}

  void CPUImageCopy::run()
  {
    assert(dst->getH() >= src->getH());
    assert(dst->getW() >= src->getW());

    ispc::CPUImageCopyKernel kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    parallel_nd(dst->getH(), [&](int h)
    {
      ispc::CPUImageCopyKernel_run(&kernel, h);
    });
  }

} // namespace oidn