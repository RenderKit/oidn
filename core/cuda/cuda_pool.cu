// Copyright 2009-2021 Intel Corporation
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

    srcDesc = toCuDNNTensor(src->getDesc());
    dstDesc = toCuDNNTensor(dst->getDesc());
  }

  CUDAPool::~CUDAPool()
  {
    checkError(cudnnDestroyPoolingDescriptor(poolDesc));
    checkError(cudnnDestroyTensorDescriptor(srcDesc));
    checkError(cudnnDestroyTensorDescriptor(dstDesc));
  }

  void CUDAPool::run()
  {
    const float alpha = 1;
    const float beta  = 0;
    checkError(cudnnPoolingForward(device->getCuDNNHandle(),
                                  poolDesc,
                                  &alpha,
                                  srcDesc,
                                  src->getData(),
                                  &beta,
                                  dstDesc,
                                  dst->getData()));
  }

} // namespace oidn