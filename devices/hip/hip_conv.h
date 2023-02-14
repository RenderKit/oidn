// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "hip_common.h"

OIDN_NAMESPACE_BEGIN

  class HIPConv final : public Conv
  {
  public:
    HIPConv(const Ref<HIPEngine>& engine, const ConvDesc& desc);
    ~HIPConv();

    bool isSupported() const override;

    size_t getScratchByteSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;

    void finalize() override;
    void submit() override;

  private:
    Ref<HIPEngine> engine;
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

OIDN_NAMESPACE_END
