// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"

namespace oidn {

  class SYCLDevice : public Device
  { 
  public:
    static bool isSupported();
    static bool isDeviceSupported(const sycl::device& device);

    SYCLDevice();
    SYCLDevice(const sycl::queue& syclQueue);

    OIDN_INLINE sycl::device&  getSYCLDevice()  { return sycl->device; }
    OIDN_INLINE sycl::context& getSYCLContext() { return sycl->context; }
    OIDN_INLINE sycl::queue&   getSYCLQueue()   { return sycl->queue; }

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
    Storage getPointerStorage(const void* ptr) override;

    // Enqueues a basic kernel
    template<int N, typename F>
    OIDN_INLINE void runKernelAsync(WorkDim<N> globalSize, const F& f)
    {
      sycl->queue.parallel_for<F>(globalSize, [=](sycl::item<N> it) { f(it); });
    }

    // Enqueues a work-group kernel
    template<int N, typename F>
    OIDN_INLINE void runKernelAsync(WorkDim<N> numGroups, WorkDim<N> groupSize, const F& f)
    {
      sycl->queue.parallel_for<F>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        [=](sycl::nd_item<N> it) { f(it); });
    }

    // Enqueues a basic ESIMD kernel
    template<int N, typename F>
    OIDN_INLINE void runESIMDKernelAsync(WorkDim<N> globalSize, const F& f)
    {
      sycl->queue.parallel_for<F>(
        globalSize,
        [=](sycl::item<N> it) SYCL_ESIMD_KERNEL { f(it); });
    }

    // Enqueues a work-group ESIMD kernel
    template<int N, typename F>
    OIDN_INLINE void runESIMDKernelAsync(WorkDim<N> numGroups, WorkDim<N> groupSize, const F& f)
    {
      sycl->queue.parallel_for<F>(
        sycl::nd_range<N>(numGroups * groupSize, groupSize),
        [=](sycl::nd_item<N> it) SYCL_ESIMD_KERNEL { f(it); });
    }

    // Enqueues a host function
    void runHostFuncAsync(std::function<void()>&& f) override;

    void wait() override;

    int getMaxWorkGroupSize() const override { return maxWorkGroupSize; }

  private:
    void init() override;

    struct SYCL
    {
      sycl::context context;
      sycl::device  device;
      sycl::queue   queue;
    };

    // GPU architecture
    enum class Arch
    {
      Gen9,
      XeHPG,
      XeHPC,
    };

    std::unique_ptr<SYCL> sycl;

    Arch arch;
    int maxWorkGroupSize = 0;
  };

} // namespace oidn
