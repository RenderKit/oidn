// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_image_copy.h"
#include "image_copy_kernel_ispc.h"

namespace oidn {

  void cpuImageCopy(const Image& src,
                    const Image& dst)
  {
    assert(dst.height >= src.height);
    assert(dst.width  >= src.width);

    ispc::ImageCopy kernel;

    kernel.src = src;
    kernel.dst = dst;

    kernel.H = dst.height;
    kernel.W = dst.width;

    parallel_nd(kernel.H, [&](int h)
    {
      ispc::ImageCopy_kernel(&kernel, h);
    });
  }

} // namespace oidn