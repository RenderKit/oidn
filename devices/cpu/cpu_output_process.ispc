// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"
#include "image_accessor.isph"
#include "color.isph"
#include "tile.isph"

struct CPUOutputProcessKernel
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

export void CPUOutputProcessKernel_run(const uniform CPUOutputProcessKernel* uniform self,
                                       uniform int h)
{
  const uniform int hSrc = h + self->tile.hSrcBegin;
  const uniform int hDst = h + self->tile.hDstBegin;

  const uniform float outputScale = TransferFunction_getOutputScale(&self->transferFunc);

  foreach (w = 0 ... self->tile.W)
  {
    const int wSrc = w + self->tile.wSrcBegin;
    const int wDst = w + self->tile.wDstBegin;

    // Load
    vec3f value = Tensor_get3(self->src, 0, hSrc, wSrc);

    // The CNN output may contain negative values or even NaNs, so it must be sanitized
    value = clamp(nan_to_zero(value), 0.f, pos_max);

    // Apply the inverse transfer function
    value = self->transferFunc.inverse(&self->transferFunc, value);

    // Average the channels if there is only one output channel
    if (self->dst.C == 1)
      value = make_vec3f((value.x + value.y + value.z) * (1.f / 3.f));

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
    value = value * outputScale;

    // Store
    Image_set3(self->dst, hDst, wDst, value);
  }
}