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

#pragma once

#include "common/platform.h"

namespace oidn {

  using std::log2;
  using std::exp2;
  using std::pow;
  using std::isfinite;

  __forceinline float rcp(float x)
  {
    __m128 r = _mm_rcp_ss(_mm_set_ss(x));
    return _mm_cvtss_f32(_mm_sub_ss(_mm_add_ss(r, r), _mm_mul_ss(_mm_mul_ss(r, r), _mm_set_ss(x))));
  }

  // Filters out NaN and inf, and optionally clamps to zero
  template<bool doClamp>
  __forceinline float sanitize(float x)
  {
    return isfinite(x) ? (doClamp ? max(x, 0.f) : x) : 0.f;
  }

} // namespace oidn
