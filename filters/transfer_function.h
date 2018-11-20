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

#include "common.h"

namespace oidn {

  class LinearTransferFunction
  {
  public:
    __forceinline float forward(float x) const { return x; }
    __forceinline float reverse(float x) const { return x; }
  };

  class SrgbTransferFunction
  {
  public:
    __forceinline float forward(float x) const { return std::pow(x, 1.f/2.2f); }
    __forceinline float reverse(float x) const { return std::pow(x, 2.2f); }
  };

  // HDR = Reinhard + sRGB
  class HdrTransferFunction
  {
  private:
    float exposure;
    float invExposure;

  public:
    HdrTransferFunction(float exposure = 1.f)
      : exposure(exposure),
        invExposure(1.f / exposure)
    {}

    __forceinline float forward(float x) const
    {
      x *= exposure;
      return std::pow(x / (1.f + x), 1.f/2.2f);
    }

    __forceinline float reverse(float x) const
    {
      const float y = std::pow(x, 2.2f);
      return (y / (1.f - y)) * invExposure;
    }
  };

} // ::oidn
