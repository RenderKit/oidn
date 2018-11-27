// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

#include "autoexposure.h"

namespace oidn {

  __forceinline float luminance(float r, float g, float b)
  {
    return 0.212671f * r + 0.715160f * g + 0.072169f * b;
  }

  float autoexposure(const Image& color)
  {
    assert(color.format == Format::Float3);
    constexpr float key = 0.18f;

    using Sum = std::pair<float, int>;

    Sum sum =
      tbb::parallel_reduce(
        tbb::blocked_range<int>(0, color.height),
        Sum(0.f, 0),
        [&](const tbb::blocked_range<int>& r, Sum sum) -> Sum
        {
          for (int h = r.begin(); h != r.end(); ++h)
          {
            for (int w = 0; w < color.width; ++w)
            {
              const float* rgb = (const float*)color.get(h, w);
              const float L = luminance(rgb[0], rgb[1], rgb[2]);

              if (L > 1e-7f)
              {
                sum.first += log2(L);
                sum.second++;
              }
            }
          }
          return sum;
        },
        [](Sum a, Sum b) -> Sum { return Sum(a.first+b.first, a.second+b.second); },
        tbb::static_partitioner()
      );

    return (sum.second > 0) ? (key / exp2(sum.first / float(sum.second))) : 1.f;
  }

} // namespace oidn
