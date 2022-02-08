// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "cuda_common.h"

namespace oidn {

  class CUDAPool : public CUDAOp, public Pool
  {
  private:
    cudnnPoolingDescriptor_t poolDesc;
    cudnnTensorDescriptor_t srcDesc;
    cudnnTensorDescriptor_t dstDesc;

  public:
    CUDAPool(const Ref<CUDADevice>& device, const PoolDesc& desc);
    ~CUDAPool();

    void run() override;
  };

} // namespace oidn
