// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"
#include "image_accessor.isph"
#include "color.isph"
#include "tile.isph"

struct CPUInputProcessKernel
{
  // Source
  uniform ImageAccessor input;  // color, albedo or normal
  uniform ImageAccessor albedo; // auxiliary albedo
  uniform ImageAccessor normal; // auxiliary normal

  // Destination
  uniform TensorAccessor3D dst;

  // Tile
  uniform Tile tile;

  // Transfer function
  uniform TransferFunction transferFunc;
  uniform bool hdr;
  uniform bool snorm; // signed normalized ([-1..1])
};

// Gets an input value
inline vec3f getInput(const uniform CPUInputProcessKernel* uniform self, uniform int h, int w)
{
  vec3f value = Image_get3(self->input, h, w);

  // Scale
  value = value * TransferFunction_getInputScale(&self->transferFunc);

  // Sanitize
  value = clamp(nan_to_zero(value), self->snorm ? -1.f : 0.f, self->hdr ? pos_max : 1.f);

  if (self->snorm)
  {
    // Transform to [0..1]
    value = value * 0.5f + 0.5f;
  }

  // Apply the transfer function
  value = self->transferFunc.forward(&self->transferFunc, value);

  return value;
}

// Gets an albedo value
inline vec3f getAlbedo(const uniform CPUInputProcessKernel* uniform self, uniform int h, int w)
{
  vec3f value = Image_get3(self->albedo, h, w);

  // Sanitize
  value = clamp(nan_to_zero(value), 0.f, 1.f);

  return value;
}

// Gets a normal value
inline vec3f getNormal(const uniform CPUInputProcessKernel* uniform self, uniform int h, int w)
{
  vec3f value = Image_get3(self->normal, h, w);

  // Sanitize
  value = clamp(nan_to_zero(value), -1.f, 1.f);

  // Transform to [0..1]
  value = value * 0.5f + 0.5f;

  return value;
}

export void CPUInputProcessKernel_run(const uniform CPUInputProcessKernel* uniform self,
                                      uniform int hDst)
{
  const uniform int h = hDst - self->tile.hDstBegin;

  if (h >= 0 && h < self->tile.H)
  {
    const uniform int hSrc = h + self->tile.hSrcBegin;

    // Zero pad
    foreach (wDst = 0 ... self->tile.wDstBegin)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        Tensor_set(self->dst, c, hDst, wDst, 0);
    }

    // Reorder
    foreach (w = 0 ... self->tile.W)
    {
      const int wSrc = w + self->tile.wSrcBegin;
      const int wDst = w + self->tile.wDstBegin;

      Tensor_set3(self->dst, 0, hDst, wDst, getInput(self, hSrc, wSrc));
      uniform int c = 3;

      if (self->albedo.ptr)
      {
        Tensor_set3(self->dst, 3, hDst, wDst, getAlbedo(self, hSrc, wSrc));
        c += 3;

        if (self->normal.ptr)
        {
          Tensor_set3(self->dst, 6, hDst, wDst, getNormal(self, hSrc, wSrc));
          c += 3;
        }
      }

      for (; c < self->dst.C; ++c)
        Tensor_set(self->dst, c, hDst, wDst, 0);
    }

    // Zero pad
    foreach (wDst = self->tile.W + self->tile.wDstBegin ... self->dst.W)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        Tensor_set(self->dst, c, hDst, wDst, 0);
    }
  }
  else
  {
    // Zero pad
    foreach (wDst = 0 ... self->dst.W)
    {
      for (uniform int c = 0; c < self->dst.C; ++c)
        Tensor_set(self->dst, c, hDst, wDst, 0);
    }
  }
}

