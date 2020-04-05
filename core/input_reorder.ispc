// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "memory.ih"
#include "image.ih"
#include "color.ih"

struct InputReorder
{
  // Source
  uniform Image srcColor;
  uniform Image srcAlbedo;
  uniform Image srcNormal;

  // Destination
  uniform Memory dst;

  // Tile
  uniform int hSrcBegin;
  uniform int wSrcBegin;
  uniform int hDstBegin;
  uniform int wDstBegin;
  uniform int H;
  uniform int W;

  // Transfer function
  uniform TransferFunction* uniform transferFunc;
  uniform bool hdr;
};

inline void storeZero(uniform InputReorder* uniform self, uniform int h, int w, uniform int c)
{
  set1f(self->dst, h, w, c, 0.f);
}

// Stores a color value
inline void storeColor(uniform InputReorder* uniform self, uniform int h, int w, uniform int c, vec3f value)
{
  // Sanitize
  value = clamp(nan_to_zero(value), 0.f, self->hdr ? pos_max : 1.f);

  // Apply the transfer function
  value = self->transferFunc->forward(self->transferFunc, value);

  // Store
  set3f(self->dst, h, w, c, value);
}

// Stores an albedo value
inline void storeAlbedo(uniform InputReorder* uniform self, uniform int h, int w, uniform int c, vec3f value)
{
  // Sanitize
  value = clamp(nan_to_zero(value), 0.f, 1.f);

  // Store
  set3f(self->dst, h, w, c, value);
}

// Stores a normal value
inline void storeNormal(uniform InputReorder* uniform self, uniform int h, int w, uniform int c, vec3f value)
{
  // Sanitize
  value = clamp(nan_to_zero(value), neg_max, pos_max);

  // Normalize
  value = normalize_safe(value);

  // Transform to [0..1]
  value = value * 0.5f + 0.5f;

  // Store
  set3f(self->dst, h, w, c, value);
}

export void InputReorder_kernel(uniform InputReorder* uniform self, uniform int hDst)
{
  const uniform int h = hDst - self->hDstBegin;

  if (h >= 0 && h < self->H)
  {
    const uniform int hSrc = h + self->hSrcBegin;

    // Zero pad
    foreach (wDst = 0 ... self->wDstBegin)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        storeZero(self, hDst, wDst, c);
    }

    // Reorder
    foreach (w = 0 ... self->W)
    {
      const int wSrc = w + self->wSrcBegin;
      const int wDst = w + self->wDstBegin;

      uniform int c = 0;
      storeColor(self, hDst, wDst, c, get3f(self->srcColor, hSrc, wSrc));
      c += 3;

      if (self->srcAlbedo.ptr)
      {
        storeAlbedo(self, hDst, wDst, c, get3f(self->srcAlbedo, hSrc, wSrc));
        c += 3;
      }

      if (self->srcNormal.ptr)
      {
        storeNormal(self, hDst, wDst, c, get3f(self->srcNormal, hSrc, wSrc));
        c += 3;
      }

      for (; c < self->dst.C; ++c)
        storeZero(self, hDst, wDst, c);
    }

    // Zero pad
    foreach (wDst = self->W + self->wDstBegin ... self->dst.W)
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

