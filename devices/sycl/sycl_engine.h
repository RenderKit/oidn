// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "sycl_device.h"
#include <optional>

OIDN_NAMESPACE_BEGIN

  class SYCLEngine : public Engine
  {
    friend class SYCLDevice;

  public:
    SYCLEngine(const Ref<SYCLDevice>& device,
               const sycl::queue& syclQueue);

    Device* getDevice() const override { return device; }
    ze_device_handle_t getZeDevice() const { return zeDevice; }

    // Buffer
    Ref<Buffer> newExternalBuffer(ExternalMemoryTypeFlag fdType,
                                  int fd, size_t byteSize) override;

    Ref<Buffer> newExternalBuffer(ExternalMemoryTypeFlag handleType,
                                  void* handle, const void* name, size_t byteSize) override;

    // Ops
    bool isConvSupported(PostOp postOp) override;
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

    // Enqueues a basic kernel
    template<int N, typename F>
    OIDN_INLINE void submitKernel(WorkDim<N> globalSize, const F& f)
    {
      lastEvent = syclQueue.parallel_for<F>(
        globalSize,
        getDepEvents(),
        [=](sycl::item<N> it) { f(it); });
    }

    // Enqueues a work-group kernel
    template<int N, typename F>
    OIDN_INLINE void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const F& f)
    {
      lastEvent = syclQueue.parallel_for<F>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        getDepEvents(),
        [=](sycl::nd_item<N> it) { f(it); });
    }

    // Enqueues a work-group kernel with explicit subgroup size
    template<int subgroupSize, int N, typename F>
    OIDN_INLINE void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const F& f)
    {
      lastEvent = syclQueue.parallel_for<F>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        getDepEvents(),
        [=](sycl::nd_item<N> it) [[intel::reqd_sub_group_size(subgroupSize)]] { f(it); });
    }

    // Enqueues a basic ESIMD kernel
    template<int N, typename F>
    OIDN_INLINE void submitESIMDKernel(WorkDim<N> globalSize, const F& f)
    {
      lastEvent = syclQueue.parallel_for<F>(
        globalSize,
        getDepEvents(),
        [=](sycl::item<N> it) SYCL_ESIMD_KERNEL { f(it); });
    }

    // Enqueues a work-group ESIMD kernel
    template<int N, typename F>
    OIDN_INLINE void submitESIMDKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const F& f)
    {
      lastEvent = syclQueue.parallel_for<F>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        getDepEvents(),
        [=](sycl::nd_item<N> it) SYCL_ESIMD_KERNEL { f(it); });
    }

    void submitHostFunc(std::function<void()>&& f) override;

    void wait() override;

    int getMaxWorkGroupSize() const override { return maxWorkGroupSize; }

  private:
    void submitBarrier();

    OIDN_INLINE std::vector<sycl::event> getDepEvents()
    {
      if (!depEvents.empty())
        return std::move(depEvents); // override the default once
      else if (lastEvent)
        return {lastEvent.value()};  // default
      else
        return {};
    }

    SYCLDevice* device;
    ze_device_handle_t zeDevice = nullptr; // Level Zero device
    sycl::queue syclQueue;                 // all commands are submitted to this queue
    std::optional<sycl::event> lastEvent;  // the last submitted command, implicit dependency of the next command
    std::vector<sycl::event> depEvents;    // explicit dependencies of the next command, if not empty

    int maxWorkGroupSize = 0;
  };

OIDN_NAMESPACE_END
