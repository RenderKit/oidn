// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "hip_common.h"

namespace oidn {

  class HIPConv final : public Conv
  {
  public:
    HIPConv(const Ref<HIPDevice>& device, const ConvDesc& desc);
    ~HIPConv();

    bool isSupported() const override;

    size_t getScratchByteSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;

    void finalize() override;
    void run() override;

  private:
    Ref<HIPDevice> device;
    bool finalized = false;

    miopenConvolutionDescriptor_t convDesc;
    miopenConvFwdAlgorithm_t algo;
    miopenActivationDescriptor_t activationDesc;
    miopenTensorDescriptor_t xDesc;
    miopenTensorDescriptor_t wDesc;
    miopenTensorDescriptor_t bDesc;
    miopenTensorDescriptor_t yDesc;

    std::shared_ptr<Tensor> scratch;
  };

} // namespace oidn
