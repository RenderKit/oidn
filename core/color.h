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
    float inputScale   = 1.f;
    float outputScale  = 1.f;

  public:
    explicit TransferFunction(Type type = Type::Linear);

    void setInputScale(float inputScale)
    {
      this->inputScale  = inputScale;
      this->outputScale = (inputScale != 0.f) ? (1.f / inputScale) : 0.f;
    }

    __forceinline float getInputScale() const
    {
      return inputScale;
    }

    __forceinline float getOutputScale() const
    {
      return outputScale;
    }

    operator ispc::TransferFunction() const;
  };

  float getAutoexposure(const Image& color);

} // namespace oidn
