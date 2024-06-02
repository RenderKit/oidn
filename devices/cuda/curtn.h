// CURTN: a nano implementation of the CUDA Runtime API on top of the Driver API
// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cuda.h>
#include <cuda_runtime.h>

namespace curtn
{
  // Unlike CUDA Runtime, CURTN requires explicit initialization before the first API call
  cudaError_t init();

  // Unlike CUDA Runtime, CURTN requires explicit initialization and cleanup of the current context
  cudaError_t initContext();
  cudaError_t cleanupContext();
}