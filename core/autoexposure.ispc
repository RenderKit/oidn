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

#include "image.ih"

inline float luminance(vec3f c)
{
  return 0.212671f * c.x + 0.715160f * c.y + 0.072169f * c.z;
}

// Returns the average luminance of the specified image tile
export uniform float getAvgLuminance(const uniform Image& color,
                                     uniform int beginH, uniform int endH,
                                     uniform int beginW, uniform int endW)
{
  float L = 0.f;

  for (uniform int h = beginH; h < endH; ++h)
  {
    foreach (w = beginW ... endW)
    {
      vec3f c = get3f(color, h, w);
      c = select(isfinite(c), max(c, 0.f), make_vec3f(0.f)); // sanitize
      L += luminance(c);
    }
  }

  return reduce_add(L) / ((endH - beginH) * (endW - beginW));
}
