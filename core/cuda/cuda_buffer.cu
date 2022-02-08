// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_buffer.h"

namespace oidn {

  void* CUDABufferAllocator::allocate(const Ref<CUDADevice>& device, size_t size, MemoryKind kind)
  {
    void* ptr;

    switch (kind)
    {
    case MemoryKind::Host:
      checkError(cudaMallocHost(&ptr, size));
      return ptr;

    case MemoryKind::Device:
      checkError(cudaMalloc(&ptr, size));
      return ptr;

    case MemoryKind::Shared:
      checkError(cudaMallocManaged(&ptr, size));
      return ptr;

    default:
      throw Exception(Error::InvalidArgument, "invalid CUDA buffer type");
    }
  }

  void CUDABufferAllocator::deallocate(const Ref<CUDADevice>& device, void* ptr, MemoryKind kind)
  {
    if (kind == MemoryKind::Host)
      checkError(cudaFreeHost(ptr));
    else
      checkError(cudaFree(ptr));
  }

} // namespace oidn
