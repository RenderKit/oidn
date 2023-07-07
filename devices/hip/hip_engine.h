// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "hip_device.h"

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_COMPILE_HIP)
  // Main kernel functions
  namespace
  {
    template<typename F>
    __global__ void basicHIPKernel(WorkDim<1> globalSize, const F f)
    {
      WorkItem<1> it(globalSize);
      if (it.getGlobalID() < it.getGlobalSize())
        f(it);
    }

    template<typename F>
    __global__ void basicHIPKernel(WorkDim<2> globalSize, const F f)
    {
      WorkItem<2> it(globalSize);
      if (it.getGlobalID<0>() < it.getGlobalSize<0>() &&
          it.getGlobalID<1>() < it.getGlobalSize<1>())
        f(it);
    }

    template<typename F>
    __global__ void basicHIPKernel(WorkDim<3> globalSize, const F f)
    {
      WorkItem<3> it(globalSize);
      if (it.getGlobalID<0>() < it.getGlobalSize<0>() &&
          it.getGlobalID<1>() < it.getGlobalSize<1>() &&
          it.getGlobalID<2>() < it.getGlobalSize<2>())
        f(it);
    }

    template<int N, typename F>
    __global__ void groupHIPKernel(const F f)
    {
      f(WorkGroupItem<N>());
    }
  }
#endif

  class HIPEngine final : public Engine
  {
  public:
    HIPEngine(const Ref<HIPDevice>& device,
              hipStream_t stream);

    Device* getDevice() const override { return device; }
    hipStream_t getHIPStream() const { return stream; }

    // Buffer
    Ref<Buffer> newExternalBuffer(ExternalMemoryTypeFlag fdType,
                                  int fd, size_t byteSize) override;

    Ref<Buffer> newExternalBuffer(ExternalMemoryTypeFlag handleType,
                                  void* handle, const void* name, size_t byteSize) override;

    // Tensor
    bool isSupported(const TensorDesc& desc) const override;

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    std::shared_ptr<ImageCopy> newImageCopy() override;

    // Unified shared memory (USM)
    void* usmAlloc(size_t byteSize, Storage storage) override;
    void usmFree(void* ptr, Storage storage) override;
    void usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;
    void submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;

  #if defined(OIDN_COMPILE_HIP)
    // Enqueues a basic kernel
    template<int N, typename F>
    OIDN_INLINE void submitKernel(WorkDim<N> globalSize, const F& f)
    {
      // TODO: improve group size computation
      WorkDim<N> groupSize = suggestWorkGroupSize(globalSize);
      WorkDim<N> numGroups = ceil_div(globalSize, groupSize);

      basicHIPKernel<<<numGroups, groupSize, 0, stream>>>(globalSize, f);
      checkError(hipGetLastError());
    }

    // Enqueues a work-group kernel
    template<int N, typename F>
    OIDN_INLINE void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const F& f)
    {
      groupHIPKernel<N><<<numGroups, groupSize, 0, stream>>>(f);
      checkError(hipGetLastError());
    }
  #endif

    // Enqueues a host function
    void submitHostFunc(std::function<void()>&& f) override;

    void wait() override;

    int getMaxWorkGroupSize() const override { return device->maxWorkGroupSize; }
    int getSubgroupSize() const override { return device->subgroupSize; }

  private:
    WorkDim<1> suggestWorkGroupSize(WorkDim<1> globalSize) { return 1024; }
    WorkDim<2> suggestWorkGroupSize(WorkDim<2> globalSize) { return {32, 32}; }
    WorkDim<3> suggestWorkGroupSize(WorkDim<3> globalSize) { return {1, 32, 32}; }

    HIPDevice* device;
    hipStream_t stream;
  };

OIDN_NAMESPACE_END
