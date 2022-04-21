// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "cuda_common.h"

namespace oidn {

  class CUDAPool final : public Pool
  {
  public:
    CUDAPool(const Ref<CUDADevice>& device, const PoolDesc& desc);
    ~CUDAPool();

    bool isSupported() const override;

    void run() override;

  private:
    Ref<CUDADevice> device;
    
    cudnnPoolingDescriptor_t poolDesc;
    cudnnTensorDescriptor_t xDesc;
    cudnnTensorDescriptor_t yDesc;
  };

} // namespace oidn
