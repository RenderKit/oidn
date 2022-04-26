// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"

struct miopenHandle;
typedef struct miopenHandle* miopenHandle_t;

namespace oidn {

#if defined(OIDN_COMPILE_HIP)
  namespace
  {
    // Helper functions for kernel execution
    template<int dims, typename F>
    __global__ void runHIPKernel(const F f)
    {
      f(WorkGroupItem<dims>());
    }

    template<typename F>
    __global__ void runHIPKernel(WorkDim<2> range, const F f)
    {
      WorkItem<2> it(range);
      if (it.getId<0>() < it.getRange<0>() &&
          it.getId<1>() < it.getRange<1>())
        f(it);
    }

    template<typename F>
    __global__ void runHIPKernel(WorkDim<3> range, const F f)
    {
      WorkItem<3> it(range);
      if (it.getId<0>() < it.getRange<0>() &&
          it.getId<1>() < it.getRange<1>() &&
          it.getId<2>() < it.getRange<2>())
        f(it);
    }
  }

  void checkError(hipError_t error);
#endif

  class HIPDevice final : public Device
  { 
  public:
    ~HIPDevice();

    OIDN_INLINE miopenHandle_t getMIOpenHandle() const { return miopenHandle; }

    void wait() override;

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    std::shared_ptr<ImageCopy> newImageCopy() override;

    // Memory
    void* malloc(size_t byteSize, Storage storage) override;
    void free(void* ptr, Storage storage) override;
    void memcpy(void* dstPtr, const void* srcPtr, size_t byteSize) override;

  #if defined(OIDN_COMPILE_HIP)
    // Enqueues a basic 2D kernel
    template<typename F>
    OIDN_INLINE void runKernelAsync(WorkDim<2> range, const F& f)
    {
      const dim3 blockDim(32, 32);
      const dim3 gridDim(ceil_div(range[1], blockDim.x), ceil_div(range[0], blockDim.y));

      runHIPKernel<<<gridDim, blockDim>>>(range, f);
      checkError(hipGetLastError());
    }

    // Enqueues a basic 3D kernel
    template<typename F>
    OIDN_INLINE void runKernelAsync(WorkDim<3> range, const F& f)
    {
      const dim3 blockDim(32, 32, 1);
      const dim3 gridDim(ceil_div(range[2], blockDim.x), ceil_div(range[1], blockDim.y), ceil_div(range[0], blockDim.z));

      runHIPKernel<<<gridDim, blockDim>>>(range, f);
      checkError(hipGetLastError());
    }

    // Enqueues a group-based 1D kernel
    template<typename F>
    OIDN_INLINE void runKernelAsync(WorkDim<1> groupRange, WorkDim<1> localRange, const F& f)
    {
      runHIPKernel<1><<<groupRange[0], localRange[0]>>>(f);
      checkError(hipGetLastError());
    }

    // Enqueues a group-based 2D kernel
    template<typename F>
    OIDN_INLINE void runKernelAsync(WorkDim<2> groupRange, WorkDim<2> localRange, const F& f)
    {
      runHIPKernel<2><<<dim3(groupRange[1], groupRange[0]), dim3(localRange[1], localRange[0])>>>(f);
      checkError(hipGetLastError());
    }
  #endif

    // Enqueues a host function
    void runHostFuncAsync(std::function<void()>&& f) override;

  protected:
    void init() override;

  private:
    miopenHandle_t miopenHandle;
  };

} // namespace oidn
