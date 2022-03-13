// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../upsample.h"
#include "../upsample_kernel.h"

namespace oidn {

  template<typename OpT, typename TensorT, TensorLayout tensorLayout>
  class XPUUpsample : public OpT, public Upsample
  {
  public:
    XPUUpsample(const Ref<typename OpT::DeviceType>& device,
                    const UpsampleDesc& desc)
      : OpT(device),
        Upsample(desc) {}

    void run() override
    {
      UpsampleKernel<TensorT, tensorLayout> kernel;
      kernel.src = *src;
      kernel.dst = *dst;

      if (tensorLayout == TensorLayout::hwc)
        this->device->runKernel(src->getH(), src->getW(), src->getC(), kernel);
      else
        this->device->runKernel(src->getC(), src->getH(), src->getW(), kernel);
    }
  };

} // namespace oidn
