// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../input_process.h"
#include "../input_process_kernel.h"

namespace oidn {

  template<typename OpT, typename TensorT, TensorLayout tensorLayout>
  class XPUInputProcess : public OpT, public InputProcess
  {
  public:
    XPUInputProcess(const Ref<typename OpT::DeviceType>& device,
                    const InputProcessDesc& desc)
      : OpT(device),
        InputProcess(desc) {}

    void run() override
    {
      switch (getInput()->getDataType())
      {
      case DataType::Float32:
        runKernel<float>();
        break;
      case DataType::Float16:
        runKernel<half>();
        break;
      default:
        assert(0);
      }
    }

  private:
    template<typename ImageT>
    void runKernel()
    {
      assert(tile.H + tile.hSrcBegin <= getInput()->getH());
      assert(tile.W + tile.wSrcBegin <= getInput()->getW());
      assert(tile.H + tile.hDstBegin <= dst->getH());
      assert(tile.W + tile.wDstBegin <= dst->getW());
      
      InputProcessKernel<ImageT, TensorT, tensorLayout> kernel;
      kernel.color  = color  ? *color  : Image();
      kernel.albedo = albedo ? *albedo : Image();
      kernel.normal = normal ? *normal : Image();
      kernel.dst = *dst;
      kernel.tile = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr = hdr;
      kernel.snorm = snorm;

      this->device->runKernel({dst->getH(), dst->getW()}, kernel);
    }
  };

} // namespace oidn
