// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_accessor.isph"
#include "color.isph"

// Returns the average luminance of the specified image bin
export uniform float autoexposureDownsample(const uniform ImageAccessor& color,
                                            uniform int beginH, uniform int endH,
                                            uniform int beginW, uniform int endW)
{
  float L = 0.f;

  for (uniform int h = beginH; h < endH; ++h)
  {
    foreach (w = beginW ... endW)
    {
      vec3f c = Image_get3(color, h, w);
      c = clamp(nan_to_zero(c), 0.f, pos_max); // sanitize
      L += luminance(c);
    }
  }

  return reduce_add(L) / ((endH - beginH) * (endW - beginW));
}
