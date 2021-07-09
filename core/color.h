// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image.h"
#include "color_ispc.h"

namespace oidn {

  class TransferFunction
  {
  public:
    enum class Type
    {
      Linear,
      SRGB,
      PU,
      Log,
    };

  private:
    Type type;
    float inputScale  = 1.f;
    float outputScale = 1.f;

  public:
    explicit TransferFunction(Type type = Type::Linear) : type(type) {}

    void setInputScale(float inputScale)
    {
      this->inputScale  = inputScale;
      this->outputScale = (inputScale != 0.f) ? (1.f / inputScale) : 0.f;
    }

    operator ispc::TransferFunction() const
    {
      ispc::TransferFunction res;

      switch (type)
      {
      case Type::Linear: ispc::LinearTransferFunction_Constructor(&res); break;
      case Type::SRGB:   ispc::SRGBTransferFunction_Constructor(&res);   break;
      case Type::PU:     ispc::PUTransferFunction_Constructor(&res);     break;
      case Type::Log:    ispc::LogTransferFunction_Constructor(&res);    break;
      default:
        assert(0);
      }

      res.inputScale  = inputScale;
      res.outputScale = outputScale;
      
      return res;
    }
  };

  float getAutoexposure(const Image& color);

} // namespace oidn
