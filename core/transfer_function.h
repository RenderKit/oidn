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

  // Color transfer function
  class TransferFunction
  {
  public:
    virtual ~TransferFunction() = default;

    virtual float forward(float x) const = 0;
    virtual float reverse(float x) const = 0;
  };

  class LinearTransferFunction : public TransferFunction
  {
  public:
    __forceinline float forward(float x) const override { return x; }
    __forceinline float reverse(float x) const override { return x; }
  };

  class SrgbTransferFunction : public TransferFunction
  {
  public:
    __forceinline float forward(float x) const override
    {
      return pow(x, 1.f/2.2f);
    }

    __forceinline float reverse(float x) const override
    {
      return pow(x, 2.2f);
    }
  };

  // HDR = Reinhard + sRGB
  class HdrTransferFunction : public TransferFunction
  {
  private:
    float exposure;
    float invExposure;

  public:
    HdrTransferFunction(float exposure = 1.f)
    {
      setExposure(exposure);
    }

    void setExposure(float exposure)
    {
      this->exposure = exposure;
      this->invExposure = 1.f / exposure;
    }

    __forceinline float forward(float x) const override
    {
      x *= exposure;
      return pow(x / (1.f + x), 1.f/2.2f);
    }

    __forceinline float reverse(float x) const override
    {
      const float y = min(pow(x, 2.2f), 0.9999999f); // must clamp to avoid infinity
      return (y / (1.f - y)) * invExposure;
    }
  };

} // namespace oidn
