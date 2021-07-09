// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor.isph"
#include "image.isph"
#include "color.isph"
#include "reorder.isph"

struct InputReorder
{
  // Source
  uniform ImageAccessor color;
  uniform ImageAccessor albedo;
  uniform ImageAccessor normal;

  // Destination
  uniform TensorAccessor dst;

  // Tile
  uniform ReorderTile tile;

  // Transfer function
  uniform TransferFunction transferFunc;
  uniform bool hdr;
  uniform bool snorm; // signed normalized ([-1..1])
};

inline void storeZero(uniform InputReorder* uniform self, uniform int h, int w, uniform int c)
{
  set1f(self->dst, h, w, c, 0.f);
}

// Stores a color value
inline void storeColor(uniform InputReorder* uniform self, uniform int h, int w, uniform int c, vec3f value)
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
  set3f(self->dst, h, w, c, value);
}

// Stores an albedo value
inline void storeAlbedo(uniform InputReorder* uniform self, uniform int h, int w, uniform int c, vec3f value)
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
  set3f(self->dst, h, w, c, value);
}

// Stores a normal value
inline void storeNormal(uniform InputReorder* uniform self, uniform int h, int w, uniform int c, vec3f value)
{
  // Scale
  if (!self->color.ptr)
    value = value * self->transferFunc.inputScale;

  // Sanitize
  value = clamp(nan_to_zero(value), -1.f, 1.f);

  // Transform to [0..1]
  value = value * 0.5f + 0.5f;

  // Store
  set3f(self->dst, h, w, c, value);
}

export void InputReorder_kernel(uniform InputReorder* uniform self, uniform int hDst)
{
  const uniform int h = hDst - self->tile.hDstBegin;

  if (h >= 0 && h < self->tile.H)
  {
    const uniform int hSrc = h + self->tile.hSrcBegin;

    // Zero pad
    foreach (wDst = 0 ... self->tile.wDstBegin)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        storeZero(self, hDst, wDst, c);
    }

    // Reorder
    foreach (w = 0 ... self->tile.W)
    {
      const int wSrc = w + self->tile.wSrcBegin;
      const int wDst = w + self->tile.wDstBegin;

      uniform int c = 0;

      if (self->color.ptr)
      {
        storeColor(self, hDst, wDst, c, get3f(self->color, hSrc, wSrc));
        c += 3;
      }

      if (self->albedo.ptr)
      {
        storeAlbedo(self, hDst, wDst, c, get3f(self->albedo, hSrc, wSrc));
        c += 3;
      }

      if (self->normal.ptr)
      {
        storeNormal(self, hDst, wDst, c, get3f(self->normal, hSrc, wSrc));
        c += 3;
      }

      for (; c < self->dst.C; ++c)
        storeZero(self, hDst, wDst, c);
    }

    // Zero pad
    foreach (wDst = self->tile.W + self->tile.wDstBegin ... self->dst.W)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        storeZero(self, hDst, wDst, c);
    }
  }
  else
  {
    // Zero pad
    foreach (wDst = 0 ... self->dst.W)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        storeZero(self, hDst, wDst, c);
    }
  }
}

