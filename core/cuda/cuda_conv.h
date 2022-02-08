// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "cuda_common.h"

namespace oidn {

  class CUDAConv : public CUDAOp, public Conv
  {
  private:
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t convAlgo;
    cudnnActivationDescriptor_t activationDesc;
    cudnnTensorDescriptor_t srcDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnTensorDescriptor_t biasDesc;
    cudnnTensorDescriptor_t dstDesc;
    std::shared_ptr<Tensor> scratch;

  public:
    CUDAConv(const Ref<CUDADevice>& device, const ConvDesc& desc);
    ~CUDAConv();

    void run() override;
    size_t getScratchSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;
  };

} // namespace oidn
