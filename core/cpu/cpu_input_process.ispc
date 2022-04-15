// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../tensor_accessor.isph"
#include "../image_accessor.isph"
#include "../color.isph"
#include "../tile.isph"

struct CPUInputProcessKernel
{
  // Source
  uniform ImageAccessor color;
  uniform ImageAccessor albedo;
  uniform ImageAccessor normal;

  // Destination
  uniform TensorAccessor3D dst;

  // Tile
  uniform Tile tile;

  // Transfer function
  uniform TransferFunction transferFunc;
  uniform bool hdr;
  uniform bool snorm; // signed normalized ([-1..1])
};

inline void storeZero(uniform CPUInputProcessKernel* uniform self, uniform int c, uniform int h, int w)
{
  set1f(self->dst, c, h, w, 0.f);
}

// Stores a color value
inline void storeColor(uniform CPUInputProcessKernel* uniform self, uniform int c, uniform int h, int w, vec3f value)
{
  // Scale
  value = value * self->transferFunc.inputScale;

  // Sanitize
  value = clamp(nan_to_zero(value), self->snorm ? -1.f : 0.f, self->hdr ? pos_max : 1.f);

  if (self->snorm)
  {
    // Transform to [0..1]
    value = value * 0.5f + 0.5f;
  }

  // Apply the transfer function
  value = self->transferFunc.forward(&self->transferFunc, value);

  // Store
  set3f(self->dst, c, h, w, value);
}

// Stores an albedo value
inline void storeAlbedo(uniform CPUInputProcessKernel* uniform self, uniform int c, uniform int h, int w, vec3f value)
{
  // Scale
  if (!self->color.ptr)
    value = value * self->transferFunc.inputScale;

  // Sanitize
  value = clamp(nan_to_zero(value), 0.f, 1.f);

  // Apply the transfer function
  if (!self->color.ptr)
    value = self->transferFunc.forward(&self->transferFunc, value);

  // Store
  set3f(self->dst, c, h, w, value);
}

// Stores a normal value
inline void storeNormal(uniform CPUInputProcessKernel* uniform self, uniform int c, uniform int h, int w, vec3f value)
{
  // Scale
  if (!self->color.ptr)
    value = value * self->transferFunc.inputScale;

  // Sanitize
  value = clamp(nan_to_zero(value), -1.f, 1.f);

  // Transform to [0..1]
  value = value * 0.5f + 0.5f;

  // Store
  set3f(self->dst, c, h, w, value);
}

export void CPUInputProcessKernel_run(uniform CPUInputProcessKernel* uniform self, uniform int hDst)
{
  const uniform int h = hDst - self->tile.hDstBegin;

  if (h >= 0 && h < self->tile.H)
  {
    const uniform int hSrc = h + self->tile.hSrcBegin;

    // Zero pad
    foreach (wDst = 0 ... self->tile.wDstBegin)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        storeZero(self, c, hDst, wDst);
    }

    // Reorder
    foreach (w = 0 ... self->tile.W)
    {
      const int wSrc = w + self->tile.wSrcBegin;
      const int wDst = w + self->tile.wDstBegin;

      uniform int c = 0;

      if (self->color.ptr)
      {
        storeColor(self, c, hDst, wDst, get3f(self->color, hSrc, wSrc));
        c += 3;
      }

      if (self->albedo.ptr)
      {
        storeAlbedo(self, c, hDst, wDst, get3f(self->albedo, hSrc, wSrc));
        c += 3;
      }

      if (self->normal.ptr)
      {
        storeNormal(self, c, hDst, wDst, get3f(self->normal, hSrc, wSrc));
        c += 3;
      }

      for (; c < self->dst.C; ++c)
        storeZero(self, c, hDst, wDst);
    }

    // Zero pad
    foreach (wDst = self->tile.W + self->tile.wDstBegin ... self->dst.W)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        storeZero(self, c, hDst, wDst);
    }
  }
  else
  {
    // Zero pad
    foreach (wDst = 0 ... self->dst.W)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        storeZero(self, c, hDst, wDst);
    }
  }
}

