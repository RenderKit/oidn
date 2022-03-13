// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_accessor.h"
#include "image_accessor.h"
#include "color.h"
#include "tile.h"

namespace oidn {

  template<typename ImageT, typename TensorT, TensorLayout tensorLayout>
  struct InputProcessKernel
  {
    // Source
    ImageAccessor<ImageT> color;
    ImageAccessor<ImageT> albedo;
    ImageAccessor<ImageT> normal;

    // Destination
    TensorAccessor3D<TensorT, tensorLayout> dst;

    // Tile
    Tile tile;

    // Transfer function
    TransferFunction transferFunc;
    bool hdr;
    bool snorm; // signed normalized ([-1..1])

    OIDN_DEVICE_INLINE void storeZero(int c, int h, int w) const
    {
      dst(c, h, w) = 0.f;
    }

    // Stores a color value
    OIDN_DEVICE_INLINE void storeColor(int c, int h, int w, vec3f value) const
    {
      // Scale
      value = value * transferFunc.getInputScale();

      // Sanitize
      value = clamp(nan_to_zero(value), snorm ? -1.f : 0.f, hdr ? FLT_MAX : 1.f);

      if (snorm)
      {
        // Transform to [0..1]
        value = value * 0.5f + 0.5f;
      }

      // Apply the transfer function
      value = transferFunc.forward(value);

      // Store
      dst.set3(c, h, w, value);
    }

    // Stores an albedo value
    OIDN_DEVICE_INLINE void storeAlbedo(int c, int h, int w, vec3f value) const
    {
      // Scale
      if (!color.ptr)
        value = value * transferFunc.getInputScale();

      // Sanitize
      value = clamp(nan_to_zero(value), 0.f, 1.f);

      // Apply the transfer function
      if (!color.ptr)
        value = transferFunc.forward(value);

      // Store
      dst.set3(c, h, w, value);
    }

    // Stores a normal value
    OIDN_DEVICE_INLINE void storeNormal(int c, int h, int w, vec3f value) const
    {
      // Scale
      if (!color.ptr)
        value = value * transferFunc.getInputScale();

      // Sanitize
      value = clamp(nan_to_zero(value), -1.f, 1.f);

      // Transform to [0..1]
      value = value * 0.5f + 0.5f;

      // Store
      dst.set3(c, h, w, value);
    }

    OIDN_DEVICE_INLINE void operator ()(int hDst, int wDst) const
    {
      const int h = hDst - tile.hDstBegin;
      const int w = wDst - tile.wDstBegin;

      if (h >= 0 && h < tile.H && w >= 0 && w < tile.W)
      {
        const int hSrc = h + tile.hSrcBegin;
        const int wSrc = w + tile.wSrcBegin;
        const int wDst = w + tile.wDstBegin;

        int c = 0;

        if (color.ptr)
        {
          storeColor(c, hDst, wDst, color.get3(hSrc, wSrc));
          c += 3;
        }

        if (albedo.ptr)
        {
          storeAlbedo(c, hDst, wDst, albedo.get3(hSrc, wSrc));
          c += 3;
        }

        if (normal.ptr)
        {
          storeNormal(c, hDst, wDst, normal.get3(hSrc, wSrc));
          c += 3;
        }

        for (; c < dst.C; ++c)
          storeZero(c, hDst, wDst);
      }
      else
      {
        // Zero pad
        for (int c = 0; c < dst.C; ++c)
          storeZero(c, hDst, wDst);
      }
    }
  };

} // namespace oidn
