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
    template<typename F>
    __global__ void runCUDAKernel(const F f)
    {
      f();
    }

    template<typename Ty, typename Tx, typename F>
    __global__ void runCUDAParallelFor(Ty Dy, Tx Dx, const F f)
    {
      const Tx x = Tx(blockDim.x * blockIdx.x + threadIdx.x);
      const Ty y = Ty(blockDim.y * blockIdx.y + threadIdx.y);
      if (y < Dy && x < Dx)
        f(y, x);
    }

    template<typename Tz, typename Ty, typename Tx, typename F>
    __global__ void runCUDAParallelFor(Tz Dz, Ty Dy, Tx Dx, const F f)
    {
      const Tx x = Tx(blockDim.x * blockIdx.x + threadIdx.x);
      const Ty y = Ty(blockDim.y * blockIdx.y + threadIdx.y);
      const Tz z = Tz(blockDim.z * blockIdx.z + threadIdx.z);
      if (z < Dz && y < Dy && x < Dx)
        f(z, y, x);
    }
  }

  void checkError(cudaError_t error);
#endif

  class CUDADevice final : public Device
  { 
  public:
    ~CUDADevice();

    OIDN_INLINE cudnnHandle_t getCuDNNHandle() const { return cudnnHandle; }

    void wait() override;

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<ConcatConv> newConcatConv(const ConcatConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;

    void imageCopy(const Image& src, const Image& dst) override;

    // Memory
    void* malloc(size_t byteSize, Storage storage) override;
    void free(void* ptr, Storage storage) override;
    void memcpy(void* dstPtr, const void* srcPtr, size_t byteSize) override;

  #if defined(OIDN_CUDA)
    template<typename F>
    OIDN_INLINE void runKernel(int groupRange, int groupSize, const F& f)
    {
      runCUDAKernel<<<groupRange, groupSize>>>(f);
      checkError(cudaGetLastError());
    }

    template<typename F>
    OIDN_INLINE void runKernel(const std::array<int, 2>& groupRange, const std::array<int, 2>& groupSize, const F& f)
    {
      runCUDAKernel<<<dim3(groupRange[1], groupRange[0]), dim3(groupSize[1], groupSize[0])>>>(f);
      checkError(cudaGetLastError());
    }

    // Runs a parallel for kernel on the device
    template<typename Ty, typename Tx, typename F>
    OIDN_INLINE void parallelFor(const Ty& Dy, const Tx& Dx, const F& f)
    {
      const dim3 blockDim(16, 16);
      const dim3 gridDim(ceil_div(Dx, blockDim.x), ceil_div(Dy, blockDim.y));

      runCUDAParallelFor<<<gridDim, blockDim>>>(Dy, Dx, f);
      checkError(cudaGetLastError());
    }

    template<typename Tz, typename Ty, typename Tx, typename F>
    OIDN_INLINE void parallelFor(const Tz& Dz, const Ty& Dy, const Tx& Dx, const F& f)
    {
      const dim3 blockDim(16, 16, 1);
      const dim3 gridDim(ceil_div(Dx, blockDim.x), ceil_div(Dy, blockDim.y), ceil_div(Dz, blockDim.z));

      runCUDAParallelFor<<<gridDim, blockDim>>>(Dz, Dy, Dx, f);
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
