// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_concat_conv.h"

namespace oidn {

  CUDAConcatConv::CUDAConcatConv(const Ref<CUDADevice>& device, const ConcatConvDesc& desc)
    : CUDAOp(device),
      ConcatConv(desc)
  {
    // Split the convolution into two smaller convolutions
    weight1Desc = {{weight->getO(), src1Desc.getC(), weight->getH(), weight->getW()}, weight->getLayout(), weight->getDataType()};
    weight2Desc = {{weight->getO(), src2Desc.getC(), weight->getH(), weight->getW()}, weight->getLayout(), weight->getDataType()};

    checkError(cudnnCreateConvolutionDescriptor(&convDesc));
    checkError(cudnnSetConvolution2dDescriptor(convDesc,
                                               1,
                                               1,
                                               1,
                                               1,
                                               1,
                                               1,
                                               CUDNN_CONVOLUTION,
                                               toCuDNN(dstDesc.dataType)));

    // Enable Tensor Core operations
    checkError(cudnnSetConvolutionMathType(convDesc,
                                           CUDNN_TENSOR_OP_MATH));

    algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    checkError(cudnnCreateActivationDescriptor(&activationDesc));
    checkError(cudnnSetActivationDescriptor(activationDesc,
                                            relu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY,
                                            CUDNN_PROPAGATE_NAN,
                                            0));

    x1Desc   = toCuDNNTensor(src1Desc);
    x2Desc   = toCuDNNTensor(src2Desc);
    w1Desc   = toCuDNNFilter(weight1Desc);
    w2Desc   = toCuDNNFilter(weight2Desc);
    biasDesc = toCuDNNTensor(bias->getDesc());
    yDesc    = toCuDNNTensor(dstDesc);
  }

  CUDAConcatConv::~CUDAConcatConv()
  {
    checkError(cudnnDestroyConvolutionDescriptor(convDesc));
    checkError(cudnnDestroyActivationDescriptor(activationDesc));
    checkError(cudnnDestroyTensorDescriptor(x1Desc));
    checkError(cudnnDestroyTensorDescriptor(x2Desc));
    checkError(cudnnDestroyFilterDescriptor(w1Desc));
    checkError(cudnnDestroyFilterDescriptor(w2Desc));
    checkError(cudnnDestroyTensorDescriptor(biasDesc));
    checkError(cudnnDestroyTensorDescriptor(yDesc));
  }

  bool CUDAConcatConv::isSupported() const
  {
    return x1Desc && x2Desc && w1Desc && w2Desc && biasDesc && yDesc;
  }

  size_t CUDAConcatConv::getScratchByteSize() const
  {
    assert(isSupported());

    size_t scratchByteSize1;
    checkError(cudnnGetConvolutionForwardWorkspaceSize(device->getCuDNNHandle(),
                                                       x1Desc,
                                                       w1Desc,
                                                       convDesc,
                                                       yDesc,
                                                       algo,
                                                       &scratchByteSize1));

    size_t scratchByteSize2;
    checkError(cudnnGetConvolutionForwardWorkspaceSize(device->getCuDNNHandle(),
                                                       x2Desc,
                                                       w2Desc,
                                                       convDesc,
                                                       yDesc,
                                                       algo,
                                                       &scratchByteSize2));
                                                  
    return max(scratchByteSize1, scratchByteSize2);
  }

  void CUDAConcatConv::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
  }

  void CUDAConcatConv::finalize()
  {
    // Split weight into weight1 and weight2
    weight1 = device->newTensor(weight1Desc);
    weight2 = device->newTensor(weight2Desc);

    auto weightHost  = weight->map(Access::Read);
    auto weight1Host = weight1->map(Access::WriteDiscard);
    auto weight2Host = weight2->map(Access::WriteDiscard);

    TensorAccessor4D<half, TensorLayout::ohwi> weightAcc  = *weightHost;
    TensorAccessor4D<half, TensorLayout::ohwi> weight1Acc = *weight1Host;
    TensorAccessor4D<half, TensorLayout::ohwi> weight2Acc = *weight2Host;

    for (int o = 0; o < weightAcc.O; ++o)
    {
      for (int h = 0; h < weightAcc.H; ++h)
      {
        for (int w = 0; w < weightAcc.W; ++w)
        {
          for (int i = 0; i < weight1Acc.I; ++i)
            weight1Acc(o, i, h, w) = weightAcc(o, i, h, w);

          for (int i = 0; i < weight2Acc.I; ++i)
            weight2Acc(o, i, h, w) = weightAcc(o, weight1Acc.I + i, h, w);
        }
      }
    }

    weight.reset();
  }

  void CUDAConcatConv::run()
  {
    assert(isSupported());

    const float alpha1 = 1;
    const float alpha2 = 0;

    // Convolution 1
    checkError(cudnnConvolutionForward(
      device->getCuDNNHandle(),
      &alpha1,
      x1Desc,
      src1->getData(),
      w1Desc,
      weight1->getData(),
      convDesc,
      algo,
      scratch ? scratch->getData() : nullptr,
      scratch ? scratch->getByteSize() : 0,
      &alpha2,
      yDesc,
      dst->getData()));

    // Convolution 2, accumulation, bias, activation
    checkError(cudnnConvolutionBiasActivationForward(device->getCuDNNHandle(),
                                                     &alpha1,
                                                     x2Desc,
                                                     src2->getData(),
                                                     w2Desc,
                                                     weight2->getData(),
                                                     convDesc,
                                                     algo,
                                                     scratch ? scratch->getData() : nullptr,
                                                     scratch ? scratch->getByteSize() : 0,
                                                     &alpha1,
                                                     yDesc,
                                                     dst->getData(),
                                                     biasDesc,
                                                     bias->getData(),
                                                     activationDesc,
                                                     yDesc,
                                                     dst->getData()));
  }

} // namespace oidn