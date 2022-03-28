// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "hip_common.h"

namespace oidn {

  class HIPPool final : public HIPOp, public Pool
  {
  public:
    HIPPool(const Ref<HIPDevice>& device, const PoolDesc& desc);
    ~HIPPool();

    bool isSupported() const override;

    void run() override;

  private:
    miopenPoolingDescriptor_t poolDesc;
    miopenTensorDescriptor_t xDesc;
    miopenTensorDescriptor_t yDesc;

    std::shared_ptr<Tensor> scratch;
  };

} // namespace oidn
