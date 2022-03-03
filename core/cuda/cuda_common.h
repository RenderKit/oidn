// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cudnn.h>
#include "../op.h"
#include "cuda_device.h"

namespace oidn {

  using CUDAOp = BaseOp<CUDADevice>;

  inline void checkError(cudnnStatus_t status)
  {
    if (status != CUDNN_STATUS_SUCCESS)
      throw Exception(Error::Unknown, cudnnGetErrorString(status));
  }

  cudnnDataType_t toCuDNN(DataType dataType);
  cudnnTensorDescriptor_t toCuDNNTensor(const TensorDesc& td);
  cudnnFilterDescriptor_t toCuDNNFilter(const TensorDesc& td);

} // namespace oidn
