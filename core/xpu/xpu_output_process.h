// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../output_process.h"
#include "../output_process_kernel.h"

namespace oidn {

  template<typename OpT, typename TensorT, TensorLayout tensorLayout>
  class XPUOutputProcess : public OpT, public OutputProcess
  {
  public:
    XPUOutputProcess(const Ref<typename OpT::DeviceType>& device, const OutputProcessDesc& desc)
      : OpT(device),
        OutputProcess(desc) {}

    void run() override
    {
      switch (dst->getDataType())
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
      assert(tile.hSrcBegin + tile.H <= src->getH());
      assert(tile.wSrcBegin + tile.W <= src->getW());
      //assert(tile.hDstBegin + tile.H <= dst->getH());
      //assert(tile.wDstBegin + tile.W <= dst->getW());

      OutputProcessKernel<ImageT, TensorT, tensorLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;
      kernel.tile = tile;
      kernel.transferFunc = *transferFunc;
      kernel.hdr = hdr;
      kernel.snorm = snorm;

      this->device->runKernel(tile.H, tile.W, kernel);
    }
  };

} // namespace oidn
