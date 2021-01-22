// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image.h"
#include "output_copy_ispc.h"

namespace oidn {

  // Output copy function
  inline void outputCopy(const Image& src,
                         const Image& dst)
  {
    assert(dst.height >= src.height);
    assert(dst.width  >= src.width);

    ispc::OutputCopy impl;

    impl.src = src;
    impl.dst = dst;

    impl.H = dst.height;
    impl.W = dst.width;

    parallel_nd(impl.H, [&](int h)
    {
      ispc::OutputCopy_kernel(&impl, h);
    });
  }

} // namespace oidn
