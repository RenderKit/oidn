// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "memory.ih"
#include "image.ih"
#include "color.ih"

struct OutputReorder
{
  // Source
  uniform Memory src;

  // Destination
  uniform Image dst;

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

export void OutputReorder_kernel(uniform OutputReorder* uniform self, uniform int h)
{
  const uniform int hSrc = h + self->hSrcBegin;
  const uniform int hDst = h + self->hDstBegin;

  foreach (w = 0 ... self->W)
  {
    const int wSrc = w + self->wSrcBegin;
    const int wDst = w + self->wDstBegin;

    // Load
    vec3f value = get3f(self->src, hSrc, wSrc, 0);

    // The CNN output may contain negative values or even NaNs, so it must be sanitized
    value = clamp(nan_to_zero(value), 0.f, pos_max);

    // Apply the inverse transfer function
    value = self->transferFunc->inverse(self->transferFunc, value);

    // Sanitize
    if (!self->hdr)
      value = min(value, 1.f);

    // Store
    set3f(self->dst, hDst, wDst, value);
  }
}