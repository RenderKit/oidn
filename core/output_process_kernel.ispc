// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"
#include "image_accessor.isph"
#include "color.isph"
#include "tile.isph"

struct OutputProcessKernel
{
  // Source
  uniform TensorAccessor3D src;

  // Destination
  uniform ImageAccessor dst;

  // Tile
  uniform Tile tile;

  // Transfer function
  uniform TransferFunction transferFunc;
  uniform bool hdr;
  uniform bool snorm; // signed normalized ([-1..1])
};

export void OutputProcessKernel_run(uniform OutputProcessKernel* uniform self, uniform int h)
{
  const uniform int hSrc = h + self->tile.hSrcBegin;
  const uniform int hDst = h + self->tile.hDstBegin;

  foreach (w = 0 ... self->tile.W)
  {
    const int wSrc = w + self->tile.wSrcBegin;
    const int wDst = w + self->tile.wDstBegin;

    // Load
    vec3f value = get3f(self->src, 0, hSrc, wSrc);

    // The CNN output may contain negative values or even NaNs, so it must be sanitized
    value = clamp(nan_to_zero(value), 0.f, pos_max);

    // Apply the inverse transfer function
    value = self->transferFunc.inverse(&self->transferFunc, value);

    // Sanitize
    if (self->snorm)
    {
      // Transform to [-1..1]
      value = value * 2.f - 1.f;
      value = max(value, -1.f);
    }
    if (!self->hdr)
      value = min(value, 1.f);

    // Scale
    value = value * self->transferFunc.outputScale;

    // Store
    set3f(self->dst, hDst, wDst, value);
  }
}