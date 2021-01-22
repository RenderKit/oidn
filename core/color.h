// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image.h"
#include "node.h"
#include "color_ispc.h"

namespace oidn {

  class TransferFunction : public RefCount
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
      default:           assert(0);
      }
    }

    void setExposure(float exposure)
    {
      ispc::TransferFunction_setExposure(&impl, exposure);
    }

    ispc::TransferFunction* getImpl()
    {
      return &impl;
    }
  };

  class AutoexposureNode : public Node
  {
  private:
    Image color;
    Ref<TransferFunction> transferFunc;

  public:
    AutoexposureNode(const Ref<Device>& device,
                     const Image& color,
                     const Ref<TransferFunction>& transferFunc)
      : Node(device),
        color(color),
        transferFunc(transferFunc)
    {}

    void execute() override
    {
      const float exposure = autoexposure(color);
      //printf("exposure = %f\n", exposure);
      transferFunc->setExposure(exposure);
    }

  private:
    static float autoexposure(const Image& color);
  };

} // namespace oidn
