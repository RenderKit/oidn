// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_accessor.h"
#include "image_accessor.h"
#include "color.h"
#include "reorder.h"

namespace oidn {

  template<typename ImageDT, typename TensorDT, TensorLayout tensorLayout>
  struct OutputReorder
  {
    // Source
    TensorAccessor3D<TensorDT, tensorLayout> src;

    // Destination
    ImageAccessor<ImageDT> output;

    // Tile
    ReorderTile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    OIDN_DEVICE_INLINE void operator ()(int h, int w) const
    {
      const int hSrc = h + tile.hSrcBegin;
      const int hDst = h + tile.hDstBegin;
      const int wSrc = w + tile.wSrcBegin;
      const int wDst = w + tile.wDstBegin;

      // Load
      vec3f value = src.get3(0, hSrc, wSrc);

      // The CNN output may contain negative values or even NaNs, so it must be sanitized
      value = clamp(nan_to_zero(value), 0.f, FLT_MAX);

      // Apply the inverse transfer function
      value = transferFunc.inverse(value);

      // Sanitize
      if (snorm)
      {
        // Transform to [-1..1]
        value = value * 2.f - 1.f;
        value = max(value, -1.f);
      }
      if (!hdr)
        value = min(value, 1.f);

      // Scale
      value = value * transferFunc.getOutputScale();

      // Store
      output.set3(hDst, wDst, value);
    }
  };

} // namespace oidn
