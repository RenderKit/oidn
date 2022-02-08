// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_conv.h"

namespace oidn {

  CUDAConv::CUDAConv(const Ref<CUDADevice>& device, const ConvDesc& desc)
    : CUDAOp(device),
      Conv(desc)
  {
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

    srcDesc    = toCuDNNTensor(src->getDesc());
    weightDesc = toCuDNNFilter(weight->getDesc());
    biasDesc   = toCuDNNTensor(bias->getDesc());
    dstDesc    = toCuDNNTensor(dst->getDesc());
  }

  CUDAConv::~CUDAConv()
  {
    checkError(cudnnDestroyConvolutionDescriptor(convDesc));
    checkError(cudnnDestroyActivationDescriptor(activationDesc));
    checkError(cudnnDestroyTensorDescriptor(srcDesc));
    checkError(cudnnDestroyFilterDescriptor(weightDesc));
    checkError(cudnnDestroyTensorDescriptor(biasDesc));
    checkError(cudnnDestroyTensorDescriptor(dstDesc));
  }

  void CUDAConv::run()
  {
    const float alpha1 = 1;
    const float alpha2 = 0;

    checkError(cudnnConvolutionBiasActivationForward(device->getCuDNNHandle(),
                                                     &alpha1,
                                                     srcDesc,
                                                     src->getData(),
                                                     weightDesc,
                                                     weight->getData(),
                                                     convDesc,
                                                     convAlgo,
                                                     scratch ? scratch->getData() : nullptr,
                                                     scratch ? scratch->getByteSize() : 0,
                                                     &alpha2,
                                                     dstDesc,
                                                     dst->getData(),
                                                     biasDesc,
                                                     bias->getData(),
                                                     activationDesc,
                                                     dstDesc,
                                                     dst->getData()));
                                                    

    /*
    checkError(cudnnConvolutionForward(
      device->getCuDNNHandle(),
      &alpha1,
      srcDesc,
      src->getData(),
      weightDesc,
      weight->getData(),
      convDesc,
      convAlgo,
      nullptr,
      0,
      &alpha2,
      dstDesc,
      dst->getData()));
      */
  }

  size_t CUDAConv::getScratchSize() const
  {
    size_t scratchSize;
    checkError(cudnnGetConvolutionForwardWorkspaceSize(device->getCuDNNHandle(),
                                                       srcDesc,
                                                       weightDesc,
                                                       convDesc,
                                                       dstDesc,
                                                       convAlgo,
                                                       &scratchSize));
    return scratchSize;
  }

  void CUDAConv::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
  }

} // namespace oidn