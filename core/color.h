// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image.h"
#include "color_ispc.h"

namespace oidn {

  class TransferFunction
  {
  private:
    ispc::TransferFunction impl;

  public:
    enum class Type
    {
      Linear,
      SRGB,
      PU,
      Log,
    };

    TransferFunction(Type type)
    {
      switch (type)
      {
      case Type::Linear: ispc::LinearTransferFunction_Constructor(&impl); break;
      case Type::SRGB:   ispc::SRGBTransferFunction_Constructor(&impl);   break;
      case Type::PU:     ispc::PUTransferFunction_Constructor(&impl);     break;
      case Type::Log:    ispc::LogTransferFunction_Constructor(&impl);    break;
      default:
        throw Exception(Error::Unknown, "invalid transfer function");
      }
    }

    void setInputScale(float inputScale)
    {
      ispc::TransferFunction_setInputScale(&impl, inputScale);
    }

    __forceinline ispc::TransferFunction* getImpl()
    {
      return &impl;
    }
  };

  float getAutoexposure(const Image& color);

} // namespace oidn
