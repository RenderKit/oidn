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
    template<typename Kernel>
    __global__ void basicHIPKernel(WorkDim<1> globalSize, const Kernel kernel)
    {
      WorkItem<1> it(globalSize);
      if (it.getGlobalID() < it.getGlobalSize())
        kernel(it);
    }

    template<typename Kernel>
    __global__ void basicHIPKernel(WorkDim<2> globalSize, const Kernel kernel)
    {
      WorkItem<2> it(globalSize);
      if (it.getGlobalID<0>() < it.getGlobalSize<0>() &&
          it.getGlobalID<1>() < it.getGlobalSize<1>())
        kernel(it);
    }

    template<typename Kernel>
    __global__ void basicHIPKernel(WorkDim<3> globalSize, const Kernel kernel)
    {
      WorkItem<3> it(globalSize);
      if (it.getGlobalID<0>() < it.getGlobalSize<0>() &&
          it.getGlobalID<1>() < it.getGlobalSize<1>() &&
          it.getGlobalID<2>() < it.getGlobalSize<2>())
        kernel(it);
    }

    template<int N, typename Kernel>
    __global__ void groupHIPKernel(const Kernel kernel)
    {
      kernel(WorkGroupItem<N>());
    }

    template<int N, typename Kernel>
    __global__ void groupLocalHIPKernel(const Kernel kernel)
    {
      __shared__ typename Kernel::Local local;
      kernel(WorkGroupItem<N>(), &local);
    }
  }
#endif

  class HIPEngine final : public Engine
  {
  public:
    HIPEngine(HIPDevice* device,
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
    Ref<Conv> newConv(const ConvDesc& desc) override;
    Ref<Pool> newPool(const PoolDesc& desc) override;
    Ref<Upsample> newUpsample(const UpsampleDesc& desc) override;
    Ref<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    Ref<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    Ref<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    Ref<ImageCopy> newImageCopy() override;

    // Unified shared memory (USM)
    void* usmAlloc(size_t byteSize, Storage storage) override;
    void usmFree(void* ptr, Storage storage) override;
    void usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;
    void submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;

  #if defined(OIDN_COMPILE_HIP)
    // Enqueues a basic kernel
    template<int N, typename Kernel>
    oidn_inline void submitKernel(WorkDim<N> globalSize, const Kernel& kernel)
    {
      // TODO: improve group size computation
      WorkDim<N> groupSize = suggestWorkGroupSize(globalSize);
      WorkDim<N> numGroups = ceil_div(globalSize, groupSize);

      basicHIPKernel<<<numGroups, groupSize, 0, stream>>>(globalSize, kernel);
      checkError(hipGetLastError());
    }

    // Enqueues a work-group kernel
    template<int N, typename Kernel>
    oidn_inline void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const Kernel& kernel,
                                  typename std::enable_if<!HasLocal<Kernel>::value, bool>::type = true)
    {
      groupHIPKernel<N><<<numGroups, groupSize, 0, stream>>>(kernel);
      checkError(hipGetLastError());
    }

    // Enqueues a work-group kernel using shared local memory
    template<int N, typename Kernel, typename Local = typename Kernel::Local>
    oidn_inline void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const Kernel& kernel)
    {
      groupLocalHIPKernel<N><<<numGroups, groupSize, 0, stream>>>(kernel);
      checkError(hipGetLastError());
    }
  #endif

    // Enqueues a host function
    void submitHostFunc(std::function<void()>&& kernel, const Ref<CancellationToken>& ct) override;

    void wait() override;

    HIPArch getArch() const { return device->arch; }
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
