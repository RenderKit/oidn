// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../dnnl/dnnl_device.h"

namespace oidn {

  class SYCLDevice : public DNNLDevice
  { 
  private:
    struct SYCL
    {
      sycl::context context;
      sycl::device  device;
      sycl::queue   queue;
    };

    std::unique_ptr<SYCL> sycl;

  public:
    SYCLDevice();
    SYCLDevice(const sycl::queue& syclQueue);

    OIDN_INLINE sycl::device&  getSYCLDevice()  { return sycl->device; }
    OIDN_INLINE sycl::context& getSYCLContext() { return sycl->context; }
    OIDN_INLINE sycl::queue&   getSYCLQueue()   { return sycl->queue; }

    // Ops
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;

    // Kernels
    void imageCopy(const Image& src, const Image& dst) override;

    // Memory
    void* malloc(size_t byteSize, Storage storage) override;
    void free(void* ptr, Storage storage) override;
    void memcpy(void* dstPtr, const void* srcPtr, size_t byteSize) override;

    template<typename F>
    OIDN_INLINE void runKernel(int groupRange, int groupSize, const F& f)
    {
      sycl->queue.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groupRange * groupSize),
                          sycl::range<1>(groupSize)),
        [=](sycl::nd_item<1> it) { f(); });
    }

    template<typename F>
    OIDN_INLINE void runKernel(const std::array<int, 2>& groupRange, const std::array<int, 2>& groupSize, const F& f)
    {
      sycl->queue.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(groupRange[0] * groupSize[0], groupRange[1] * groupSize[1]),
                          sycl::range<2>(groupSize[0], groupSize[1])),
        [=](sycl::nd_item<2> it) { f(); });
    }

    // Runs a kernel on the device
    template<typename Ty, typename Tx, typename F>
    OIDN_INLINE void parallelFor(const Ty& Dy, const Tx& Dx, const F& f)
    {
      sycl->queue.parallel_for(sycl::range<2>(Dy, Dx), [=](sycl::id<2> idx)
      {
        f(Ty(idx[0]), Tx(idx[1]));
      });
    }

    // Runs an ESIMD kernel on the device
    template<typename Ty, typename Tx, typename F>
    OIDN_INLINE void parallelForESIMD(const Ty& Dy, const Tx& Dx, const F& f)
    {
      // FIXME: Named kernel is necessary due to an ESIMD bug
      sycl->queue.parallel_for<class ESIMDKernel>(sycl::range<2>(Dy, Dx), [=](sycl::id<2> idx) SYCL_ESIMD_KERNEL
      {
        f(Ty(idx[0]), Tx(idx[1]));
      });
    }

  protected:
    void init() override;
    void printInfo() override;
  };

} // namespace oidn
