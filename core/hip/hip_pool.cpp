// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_pool.h"

namespace oidn {

  HIPPool::HIPPool(const Ref<HIPDevice>& device, const PoolDesc& desc)
    : HIPOp(device),
      Pool(desc)
  {
    checkError(miopenCreatePoolingDescriptor(&poolDesc));
    checkError(miopenSet2dPoolingDescriptor(poolDesc,
                                            miopenPoolingMax,
                                            2,
                                            2,
                                            0,
                                            0,
                                            2,
                                            2));

    xDesc = toMIOpen(srcDesc);
    yDesc = toMIOpen(dstDesc);
  }

  HIPPool::~HIPPool()
  {
    checkError(miopenDestroyPoolingDescriptor(poolDesc));
    checkError(miopenDestroyTensorDescriptor(xDesc));
    checkError(miopenDestroyTensorDescriptor(yDesc));
  }

  bool HIPPool::isSupported() const
  {
    return xDesc && yDesc;
  }

  void HIPPool::run()
  {
    const float alpha = 1;
    const float beta  = 0;

    checkError(miopenPoolingForward(device->getMIOpenHandle(),
                                    poolDesc,
                                    &alpha,
                                    xDesc,
                                    src->getData(),
                                    &beta,
                                    yDesc,
                                    dst->getData(),
                                    false,
                                    nullptr,
                                    0));
  }

} // namespace oidn