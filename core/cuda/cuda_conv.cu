// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_conv.h"

namespace oidn {

  CUDAConv::CUDAConv(const Ref<CUDADevice>& device, const ConvDesc& desc)
    : Conv(desc),
      device(device)
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

    xDesc    = toCuDNNTensor(srcDesc);
    wDesc    = toCuDNNFilter(weight->getDesc());
    biasDesc = toCuDNNTensor(bias->getDesc());
    yDesc    = toCuDNNTensor(dstDesc);
  }

  CUDAConv::~CUDAConv()
  {
    checkError(cudnnDestroyConvolutionDescriptor(convDesc));
    checkError(cudnnDestroyActivationDescriptor(activationDesc));
    checkError(cudnnDestroyTensorDescriptor(xDesc));
    checkError(cudnnDestroyFilterDescriptor(wDesc));
    checkError(cudnnDestroyTensorDescriptor(biasDesc));
    checkError(cudnnDestroyTensorDescriptor(yDesc));
  }

  bool CUDAConv::isSupported() const
  {
    return xDesc && wDesc && biasDesc && yDesc;
  }

  size_t CUDAConv::getScratchByteSize() const
  {
    assert(isSupported());
    
    size_t scratchByteSize;
    checkError(cudnnGetConvolutionForwardWorkspaceSize(device->getCuDNNHandle(),
                                                       xDesc,
                                                       wDesc,
                                                       convDesc,
                                                       yDesc,
                                                       algo,
                                                       &scratchByteSize));
    return scratchByteSize;
  }

  void CUDAConv::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
  }

  void CUDAConv::run()
  {
    assert(isSupported());

    const float alpha1 = 1;
    const float alpha2 = 0;

    checkError(cudnnConvolutionBiasActivationForward(device->getCuDNNHandle(),
                                                     &alpha1,
                                                     xDesc,
                                                     src->getData(),
                                                     wDesc,
                                                     weight->getData(),
                                                     convDesc,
                                                     algo,
                                                     scratch ? scratch->getData() : nullptr,
                                                     scratch ? scratch->getByteSize() : 0,
                                                     &alpha2,
                                                     yDesc,
                                                     dst->getData(),
                                                     biasDesc,
                                                     bias->getData(),
                                                     activationDesc,
                                                     yDesc,
                                                     dst->getData()));
                                                    

    /*
    checkError(cudnnConvolutionForward(
      device->getCuDNNHandle(),
      &alpha1,
      xDesc,
      src->getData(),
      wDesc,
      weight->getData(),
      convDesc,
      algo,
      nullptr,
      0,
      &alpha2,
      yDesc,
      dst->getData()));
      */
  }

} // namespace oidn