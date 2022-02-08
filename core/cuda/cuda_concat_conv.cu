// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_concat_conv.h"

namespace oidn {

  CUDAConcatConv::CUDAConcatConv(const Ref<CUDADevice>& device, const ConcatConvDesc& desc)
    : CUDAOp(device),
      ConcatConv(desc)
  {
    weight1 = device->newTensor({{desc.weight->getO(), src1->getC(), desc.weight->getH(), desc.weight->getW()}, desc.weight->getLayout(), desc.weight->getDataType()});
    weight2 = device->newTensor({{desc.weight->getO(), src2->getC(), desc.weight->getH(), desc.weight->getW()}, desc.weight->getLayout(), desc.weight->getDataType()});

    TensorAccessor4D<half, TensorLayout::ohwi> weightAcc = *desc.weight;
    TensorAccessor4D<half, TensorLayout::ohwi> weight1Acc = *weight1;
    TensorAccessor4D<half, TensorLayout::ohwi> weight2Acc = *weight2;

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

    checkError(cudnnCreateConvolutionDescriptor(&convDesc));
    checkError(cudnnSetConvolution2dDescriptor(convDesc,
                                               1,
                                               1,
                                               1,
                                               1,
                                               1,
                                               1,
                                               CUDNN_CONVOLUTION,
                                               CUDNN_DATA_HALF));

    // Enable Tensor Core operations
    checkError(cudnnSetConvolutionMathType(convDesc,
                                           CUDNN_TENSOR_OP_MATH));

    convAlgo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;

    checkError(cudnnCreateActivationDescriptor(&activationDesc));
    checkError(cudnnSetActivationDescriptor(activationDesc,
                                            desc.relu ? CUDNN_ACTIVATION_RELU : CUDNN_ACTIVATION_IDENTITY,
                                            CUDNN_PROPAGATE_NAN,
                                            0.));

    src1Desc    = toCuDNNTensor(src1->getDesc());
    src2Desc    = toCuDNNTensor(src2->getDesc());
    weight1Desc = toCuDNNFilter(weight1->getDesc());
    weight2Desc = toCuDNNFilter(weight2->getDesc());
    biasDesc    = toCuDNNTensor(bias->getDesc());
    dstDesc     = toCuDNNTensor(dst->getDesc());
  }

  CUDAConcatConv::~CUDAConcatConv()
  {
    checkError(cudnnDestroyConvolutionDescriptor(convDesc));
    checkError(cudnnDestroyActivationDescriptor(activationDesc));
    checkError(cudnnDestroyTensorDescriptor(src1Desc));
    checkError(cudnnDestroyTensorDescriptor(src2Desc));
    checkError(cudnnDestroyFilterDescriptor(weight1Desc));
    checkError(cudnnDestroyFilterDescriptor(weight2Desc));
    checkError(cudnnDestroyTensorDescriptor(biasDesc));
    checkError(cudnnDestroyTensorDescriptor(dstDesc));
  }

  void CUDAConcatConv::run()
  {
    const float alpha1 = 1;
    const float alpha2 = 0;

    checkError(cudnnConvolutionForward(
      device->getCuDNNHandle(),
      &alpha1,
      src1Desc,
      src1->getData(),
      weight1Desc,
      weight1->getData(),
      convDesc,
      convAlgo,
      scratch ? scratch->getData() : nullptr,
      scratch ? scratch->getByteSize() : 0,
      &alpha2,
      dstDesc,
      dst->getData()));

    checkError(cudnnConvolutionBiasActivationForward(device->getCuDNNHandle(),
                                                     &alpha1,
                                                     src2Desc,
                                                     src2->getData(),
                                                     weight2Desc,
                                                     weight2->getData(),
                                                     convDesc,
                                                     convAlgo,
                                                     scratch ? scratch->getData() : nullptr,
                                                     scratch ? scratch->getByteSize() : 0,
                                                     &alpha1,
                                                     dstDesc,
                                                     dst->getData(),
                                                     biasDesc,
                                                     bias->getData(),
                                                     activationDesc,
                                                     dstDesc,
                                                     dst->getData()));
  }

  size_t CUDAConcatConv::getScratchSize() const
  {
    size_t scratchSize1;
    checkError(cudnnGetConvolutionForwardWorkspaceSize(device->getCuDNNHandle(),
                                                       src1Desc,
                                                       weight1Desc,
                                                       convDesc,
                                                       dstDesc,
                                                       convAlgo,
                                                       &scratchSize1));

    size_t scratchSize2;
    checkError(cudnnGetConvolutionForwardWorkspaceSize(device->getCuDNNHandle(),
                                                      src2Desc,
                                                      weight2Desc,
                                                      convDesc,
                                                      dstDesc,
                                                      convAlgo,
                                                      &scratchSize2));
                                                  
    return max(scratchSize1, scratchSize2);
  }

  void CUDAConcatConv::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
  }

} // namespace oidn