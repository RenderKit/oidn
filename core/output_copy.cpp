// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "output_copy.h"
#include "output_copy_ispc.h"

namespace oidn {

  namespace
  {
    // Output copy function
    void cpuOutputCopy(const Image& src,
                       const Image& dst)
    {
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
  }

  void outputCopy(const Ref<Device>& device,
                  const Image& src,
                  const Image& dst)
  {
    assert(dst.height >= src.height);
    assert(dst.width  >= src.width);

    cpuOutputCopy(src, dst);
  }

} // namespace oidn