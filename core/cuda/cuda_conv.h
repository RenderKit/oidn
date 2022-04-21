// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "cuda_common.h"

namespace oidn {

  class CUDAConv final : public Conv
  {
  public:
    CUDAConv(const Ref<CUDADevice>& device, const ConvDesc& desc);
    ~CUDAConv();

    bool isSupported() const override;

    size_t getScratchByteSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;

    void run() override;

  private:
    Ref<CUDADevice> device;
    
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;
    cudnnActivationDescriptor_t activationDesc;
    cudnnTensorDescriptor_t xDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnTensorDescriptor_t biasDesc;
    cudnnTensorDescriptor_t yDesc;

    std::shared_ptr<Tensor> scratch;
  };

} // namespace oidn
