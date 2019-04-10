// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
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

#include "image.h"

namespace oidn {

  __forceinline float luminance(float r, float g, float b)
  {
    return 0.212671f * r + 0.715160f * g + 0.072169f * b;
  }

  // Color transfer function
  class TransferFunction
  {
  public:
    virtual ~TransferFunction() = default;

    virtual float forward(float y) const = 0;
    virtual float inverse(float x) const = 0;
  };

  class LinearTransferFunction : public TransferFunction
  {
  public:
    __forceinline float forward(float y) const override { return y; }
    __forceinline float inverse(float x) const override { return x; }
  };

  // LDR transfer function: sRGB curve
  class LDRTransferFunction : public TransferFunction
  {
  public:
    __forceinline float forward(float y) const override
    {
      return pow(y, 1.f/2.2f);
    }

    __forceinline float inverse(float x) const override
    {
      return pow(x, 2.2f);
    }
  };

  // HDR transfer function: PQX curve
  // Compresses [0..65504] to [0..1]
  class HDRTransferFunction : public TransferFunction
  {
  private:
    static constexpr float m1 = 2610.f / 4096.f / 4.f;
    static constexpr float m2 = 2523.f / 4096.f * 128.f;
    static constexpr float c1 = 3424.f / 4096.f;
    static constexpr float c2 = 2413.f / 4096.f * 32.f;
    static constexpr float c3 = 2392.f / 4096.f * 32.f;
    static constexpr float  a = 3711.f / 4096.f / 8.f;

    static constexpr float yScale = 80.f / 10000.f;
    static const float     xScale;

    float exposure;
    float rcpExposure;

  public:
    HDRTransferFunction(float exposure = 1.f)
    {
      setExposure(exposure);
    }

    void setExposure(float exposure)
    {
      this->exposure = exposure;
      this->rcpExposure = 1.f / exposure;
    }

    __forceinline float forward(float y) const override
    {
      y *= exposure;
      return pqx_forward(y * yScale) * xScale;
    }

    __forceinline float inverse(float x) const override
    {
      return pqx_inverse(x * (1.f/xScale)) * (1.f/yScale) * rcpExposure;
    }

  private:
    static __forceinline float pq_forward(float y)
    {
      const float yPow = pow(y, m1);
      return pow((c1 + c2 * yPow) * rcp(1.f + c3 * yPow), m2);
    }

    static __forceinline float pqx_forward(float y)
    {
      if (y <= 1.f)
        return pq_forward(y);
      else
        return a * log(y) + 1.f;
    }

    static __forceinline float pq_inverse(float x)
    {
      const float xPow = pow(x, 1.f/m2);
      return pow(max((xPow - c1) * rcp(c2 - c3 * xPow), 0.f), 1.f/m1);
    }

    static __forceinline float pqx_inverse(float x)
    {
      if (x <= 1.f)
        return pq_inverse(x);
      else
        return exp((x - 1.f) * (1.f/a));
    }
  };

  float autoexposure(const Image& color);

} // namespace oidn
