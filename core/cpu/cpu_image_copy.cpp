// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_image_copy.h"
#include "image_copy_kernel_ispc.h"

namespace oidn {

  void cpuImageCopy(const Image& src,
                    const Image& dst)
  {
    assert(dst.getH() >= src.getH());
    assert(dst.getW() >= src.getW());

    ispc::ImageCopyKernel kernel;
    kernel.src = src;
    kernel.dst = dst;

    parallel_nd(dst.getH(), [&](int h)
    {
      ispc::ImageCopyKernel_run(&kernel, h);
    });
  }

} // namespace oidn