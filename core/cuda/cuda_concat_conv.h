// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../concat_conv.h"
#include "cuda_common.h"

namespace oidn {

  class CUDAConcatConv : public CUDAOp, public ConcatConv
  {
  private:
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t convAlgo;
    cudnnActivationDescriptor_t activationDesc;
    cudnnTensorDescriptor_t src1Desc;
    cudnnTensorDescriptor_t src2Desc;
    cudnnFilterDescriptor_t weight1Desc;
    cudnnFilterDescriptor_t weight2Desc;
    cudnnTensorDescriptor_t biasDesc;
    cudnnTensorDescriptor_t dstDesc;
    std::shared_ptr<Tensor> scratch;

  public:
    CUDAConcatConv(const Ref<CUDADevice>& device, const ConcatConvDesc& desc);
    ~CUDAConcatConv();

    void run() override;
    size_t getScratchSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;
  };

} // namespace oidn
