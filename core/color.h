// =============================================================================
// Copyright 2009-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#pragma once

#include "image.h"
#include "node.h"
#include "color_ispc.h"

namespace oidn {

  class TransferFunction
  {
  private:
    ispc::TransferFunction data;

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
      case Type::Linear: ispc::LinearTransferFunction_Constructor(&data); break;
      case Type::SRGB:   ispc::SRGBTransferFunction_Constructor(&data);   break;
      case Type::PU:     ispc::PUTransferFunction_Constructor(&data);     break;
      case Type::Log:    ispc::LogTransferFunction_Constructor(&data);    break;
      default:           assert(0);
      }
    }

    void setExposure(float exposure)
    {
      ispc::TransferFunction_setExposure(&data, exposure);
    }

    ispc::TransferFunction* getIspc()
    {
      return &data;
    }
  };

  class AutoexposureNode : public Node
  {
  private:
    Image color;
    std::shared_ptr<TransferFunction> transferFunc;

  public:
    AutoexposureNode(const Image& color,
                     const std::shared_ptr<TransferFunction>& transferFunc)
      : color(color),
        transferFunc(transferFunc)
    {}

    void execute(stream& sm) override
    {
      const float exposure = autoexposure(color);
      //printf("exposure = %f\n", exposure);
      transferFunc->setExposure(exposure);
    }

  private:
    static float autoexposure(const Image& color);
  };

} // namespace oidn
