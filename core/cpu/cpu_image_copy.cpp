// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_image_copy.h"
#include "cpu_image_copy_ispc.h"

namespace oidn {

  CPUImageCopy::CPUImageCopy(const Ref<CPUDevice>& device)
    : device(device) {}

  void CPUImageCopy::run()
  {
    if (!src || !dst)
      throw std::logic_error("image copy source/destination not set");
    if (dst->getH() < src->getH() || dst->getW() < src->getW())
      throw std::out_of_range("image copy destination smaller than the source");

    ispc::CPUImageCopyKernel kernel;
    kernel.src = *src;
    kernel.dst = *dst;

    parallel_nd(dst->getH(), [&](int h)
    {
      ispc::CPUImageCopyKernel_run(&kernel, h);
    });
  }

} // namespace oidn