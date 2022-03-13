// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_pool.h"

namespace oidn {

  CUDAPool::CUDAPool(const Ref<CUDADevice>& device, const PoolDesc& desc)
    : CUDAOp(device),
      Pool(desc)
  {
    checkError(cudnnCreatePoolingDescriptor(&poolDesc));
    checkError(cudnnSetPooling2dDescriptor(poolDesc,
                                           CUDNN_POOLING_MAX,
                                           CUDNN_PROPAGATE_NAN,
                                           2,
                                           2,
                                           0,
                                           0,
                                           2,
                                           2));

    xDesc = toCuDNNTensor(srcDesc);
    yDesc = toCuDNNTensor(dstDesc);
  }

  CUDAPool::~CUDAPool()
  {
    checkError(cudnnDestroyPoolingDescriptor(poolDesc));
    checkError(cudnnDestroyTensorDescriptor(xDesc));
    checkError(cudnnDestroyTensorDescriptor(yDesc));
  }

  bool CUDAPool::isSupported() const
  {
    return xDesc && yDesc;
  }

  void CUDAPool::run()
  {
    const float alpha = 1;
    const float beta  = 0;

    checkError(cudnnPoolingForward(device->getCuDNNHandle(),
                                   poolDesc,
                                   &alpha,
                                   xDesc,
                                   src->getData(),
                                   &beta,
                                   yDesc,
                                   dst->getData()));
  }

} // namespace oidn