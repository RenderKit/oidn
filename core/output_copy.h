// Copyright 2009-2020 Intel Corporation
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

    ispc::OutputCopy data;

    data.src = toIspc(src);
    data.dst = toIspc(dst);

    data.H = dst.height;
    data.W = dst.width;

    parallel_nd(data.H, [&](int h)
    {
      ispc::OutputCopy_kernel(&data, h);
    });
  }

} // namespace oidn
