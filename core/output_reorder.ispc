// ======================================================================== //
// Copyright 2009-2020 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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

    // Load the value
    vec3f value = get3f(self->src, hSrc, wSrc, 0);

    // The CNN output may contain negative values or even NaNs, so it must be sanitized
    value = max(inf_to_zero(value), 0.f);

    // Apply the inverse transfer function
    value = self->transferFunc->inverse(self->transferFunc, value);

    // Sanitize the value
    value = max(value, 0.f);
    if (!self->hdr)
      value = min(value, 1.f);

    // Store the final value
    set3f(self->dst, hDst, wDst, value);
  }
}