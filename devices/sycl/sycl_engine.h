// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "sycl_device.h"
#include <optional>

#if __LIBSYCL_MAJOR_VERSION >= 7
  #include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#else
  #include <sycl/ext/intel/experimental/kernel_properties.hpp>
#endif

OIDN_NAMESPACE_BEGIN

  class SYCLEngine : public Engine
  {
    friend class SYCLDevice;

  public:
    SYCLEngine(SYCLDevice* device,
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

    // Enqueues a basic kernel
    template<int N, typename Kernel>
    oidn_inline void submitKernel(WorkDim<N> globalSize, const Kernel& kernel)
    {
      lastEvent = syclQueue.parallel_for<Kernel>(
        globalSize,
        getDepEvents(),
        [=](sycl::item<N> it) { kernel(it); });
    }

    // Enqueues a work-group kernel
    template<int N, typename Kernel>
    oidn_inline void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const Kernel& kernel,
                                  typename std::enable_if<!HasLocal<Kernel>::value, bool>::type = true)
    {
      lastEvent = syclQueue.parallel_for<Kernel>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        getDepEvents(),
        [=](sycl::nd_item<N> it) { kernel(it); });
    }

    // Enqueues a work-group kernel using shared local memory
    template<int N, typename Kernel, typename Local = typename Kernel::Local>
    oidn_inline void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const Kernel& kernel)
    {
      lastEvent = syclQueue.parallel_for<Kernel>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        getDepEvents(),
        [=](sycl::nd_item<N> it)
        {
          kernel(it, sycl::ext::oneapi::group_local_memory<Local>(it.get_group()));
        });
    }

    // Enqueues a work-group kernel with explicit subgroup size
    template<int subgroupSize, int N, typename Kernel>
    oidn_inline void submitKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const Kernel& kernel)
    {
      lastEvent = syclQueue.parallel_for<Kernel>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        getDepEvents(),
        [=](sycl::nd_item<N> it) [[intel::reqd_sub_group_size(subgroupSize)]] { kernel(it); });
    }

    // Enqueues a basic ESIMD kernel
    template<int N, typename Kernel>
    oidn_inline void submitESIMDKernel(WorkDim<N> globalSize, const Kernel& kernel)
    {
      lastEvent = syclQueue.parallel_for<Kernel>(
        globalSize,
        getDepEvents(),
        [=](sycl::item<N> it) SYCL_ESIMD_KERNEL { kernel(it); });
    }

    // Enqueues a work-group ESIMD kernel
    template<int N, typename Kernel>
    oidn_inline void submitESIMDKernel(WorkDim<N> numGroups, WorkDim<N> groupSize, const Kernel& kernel)
    {
      lastEvent = syclQueue.parallel_for<Kernel>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        getDepEvents(),
        [=](sycl::nd_item<N> it) SYCL_ESIMD_KERNEL { kernel(it); });
    }

    // Enqueues a work-group ESIMD kernel with large GRF
    template<int N, typename F>
    oidn_inline void submitESIMDKernelWithLargeGRF(WorkDim<N> numGroups, WorkDim<N> groupSize, const F& f)
    {
    #if __LIBSYCL_MAJOR_VERSION >= 7
      sycl::ext::oneapi::experimental::properties props{sycl::ext::intel::experimental::grf_size<256>};
      lastEvent = syclQueue.parallel_for<F>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        getDepEvents(),
        props,
        [=](sycl::nd_item<N> it) SYCL_ESIMD_KERNEL { f(it); });
    #else
      lastEvent = syclQueue.parallel_for<F>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        getDepEvents(),
        [=](sycl::nd_item<N> it) SYCL_ESIMD_KERNEL
        {
          syclx::set_kernel_properties(syclx::kernel_properties::use_large_grf);
          f(it);
        });
    #endif
    }

    void submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct) override;

    void wait() override;

    SYCLArch getArch() const { return device->getArch(); }
    int getMaxWorkGroupSize() const override { return maxWorkGroupSize; }

  private:
    void submitBarrier();

    oidn_inline std::vector<sycl::event> getDepEvents()
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
