// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cudnn.h>
#include "../op.h"
#include "cuda_device.h"

namespace oidn {

  using CUDAOp = BaseOp<CUDADevice>;

  void checkError(cudnnStatus_t status);

  // Convert data type to cuDNN equivalent
  cudnnDataType_t toCuDNN(DataType dataType);

  // Converts tensor descriptor to cuDNN equivalent
  // Returns nullptr if tensor dimensions exceed limit supported by cuDNN
  cudnnTensorDescriptor_t toCuDNNTensor(const TensorDesc& td);
  cudnnFilterDescriptor_t toCuDNNFilter(const TensorDesc& td);

} // namespace oidn
