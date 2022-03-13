// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../concat_conv.h"
#include "cuda_common.h"

namespace oidn {

  class CUDAConcatConv final : public CUDAOp, public ConcatConv
  {
  public:
    CUDAConcatConv(const Ref<CUDADevice>& device, const ConcatConvDesc& desc);
    ~CUDAConcatConv();

    bool isSupported() const override;

    size_t getScratchByteSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;

    void finalize() override;
    void run() override;

  private:
    TensorDesc weight1Desc;
    TensorDesc weight2Desc;

    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t algo;
    cudnnActivationDescriptor_t activationDesc;
    cudnnTensorDescriptor_t x1Desc;
    cudnnTensorDescriptor_t x2Desc;
    cudnnFilterDescriptor_t w1Desc;
    cudnnFilterDescriptor_t w2Desc;
    cudnnTensorDescriptor_t biasDesc;
    cudnnTensorDescriptor_t yDesc;

    std::shared_ptr<Tensor> weight1;
    std::shared_ptr<Tensor> weight2;
    std::shared_ptr<Tensor> scratch;
  };

} // namespace oidn
