// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"

struct cudnnContext;
typedef struct cudnnContext* cudnnHandle_t;

namespace oidn {

#if defined(OIDN_CUDA)
  namespace
  {
    // Helper functions for kernel execution
    template<typename Ty, typename Tx, typename F>
    __global__ void runCUDAKernel(Ty Dy, Tx Dx, const F f)
    {
      const Tx x = Tx(blockDim.x * blockIdx.x + threadIdx.x);
      const Ty y = Ty(blockDim.y * blockIdx.y + threadIdx.y);
      if (y < Dy && x < Dx)
        f(y, x);
    }

    template<typename Tz, typename Ty, typename Tx, typename F>
    __global__ void runCUDAKernel(Tz Dz, Ty Dy, Tx Dx, const F f)
    {
      const Tx x = Tx(blockDim.x * blockIdx.x + threadIdx.x);
      const Ty y = Ty(blockDim.y * blockIdx.y + threadIdx.y);
      const Tz z = Tz(blockDim.z * blockIdx.z + threadIdx.z);
      if (z < Dz && y < Dy && x < Dx)
        f(z, y, x);
    }
  }

  inline void checkError(cudaError_t error)
  {
    if (error != cudaSuccess)
      throw Exception(Error::Unknown, cudaGetErrorString(error));
  }
#endif

  class CUDADevice final : public Device
  { 
  public:
    CUDADevice();
    ~CUDADevice();

    OIDN_INLINE cudnnHandle_t getCuDNNHandle() const { return cudnnHandle; }

    void wait() override;

    Ref<Buffer> newBuffer(size_t byteSize, MemoryKind kind) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<ConcatConv> newConcatConv(const ConcatConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;

    void imageCopy(const Image& src, const Image& dst) override;

  #if defined(OIDN_CUDA)
    // Runs a kernel on the device
    template<typename Ty, typename Tx, typename F>
    OIDN_INLINE void runKernel(const Ty& Dy, const Tx& Dx, const F& f)
    {
      const dim3 blockDim(16, 16);
      const dim3 gridDim(ceil_div(Dx, blockDim.x), ceil_div(Dy, blockDim.y));

      runCUDAKernel<<<gridDim, blockDim>>>(Dy, Dx, f);
      checkError(cudaGetLastError());
    }

    template<typename Tz, typename Ty, typename Tx, typename F>
    OIDN_INLINE void runKernel(const Tz& Dz, const Ty& Dy, const Tx& Dx, const F& f)
    {
      const dim3 blockDim(16, 16, 1);
      const dim3 gridDim(ceil_div(Dx, blockDim.x), ceil_div(Dy, blockDim.y), ceil_div(Dz, blockDim.z));

      runCUDAKernel<<<gridDim, blockDim>>>(Dz, Dy, Dx, f);
      checkError(cudaGetLastError());
    }
  #endif

  protected:
    void init() override;
    void printInfo() override;

  private:
    cudnnHandle_t cudnnHandle;
  };

} // namespace oidn
