// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_accessor.h"

namespace oidn {

  template<typename TensorT, TensorLayout tensorLayout>
  struct UpsampleKernel
  {
    TensorAccessor3D<TensorT, tensorLayout> src;
    TensorAccessor3D<TensorT, tensorLayout> dst;

    OIDN_DEVICE_INLINE void operator ()(int c, int h, int w) const
    {
      const TensorT x = src(c, h, w);

      dst(c, h*2,   w*2)   = x;
      dst(c, h*2,   w*2+1) = x;
      dst(c, h*2+1, w*2)   = x;
      dst(c, h*2+1, w*2+1) = x;
    }
  };

  // Optimized for HWC layout (memory coalescing)
  template<typename TensorT>
  struct UpsampleKernel<TensorT, TensorLayout::hwc>
  {
    TensorAccessor3D<TensorT, TensorLayout::hwc> src;
    TensorAccessor3D<TensorT, TensorLayout::hwc> dst;

    OIDN_DEVICE_INLINE void operator ()(int h, int w, int c) const
    {
      const TensorT x = src(c, h, w);

      dst(c, h*2,   w*2)   = x;
      dst(c, h*2,   w*2+1) = x;
      dst(c, h*2+1, w*2)   = x;
      dst(c, h*2+1, w*2+1) = x;
    }
  };

} // namespace oidn
