// Copyright 2009-2021 Intel Corporation
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

    Ref<Buffer> newBuffer(size_t byteSize, Buffer::Kind kind) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;

    OIDN_INLINE sycl::device&  getSYCLDevice()  { return sycl->device; }
    OIDN_INLINE sycl::context& getSYCLContext() { return sycl->context; }
    OIDN_INLINE sycl::queue&   getSYCLQueue()   { return sycl->queue; }

    // Nodes
    std::shared_ptr<PoolNode> newPoolNode(const PoolDesc& desc) override;
    std::shared_ptr<UpsampleNode> newUpsampleNode(const UpsampleDesc& desc) override;
    std::shared_ptr<InputReorderNode> newInputReorderNode(const InputReorderDesc& desc) override;
    std::shared_ptr<OutputReorderNode> newOutputReorderNode(const OutputReorderDesc& desc) override;

    // Kernels
    void imageCopy(const Image& src, const Image& dst) override;

    // Executes kernel on the device
    template <typename T0, typename T1, typename F>
    OIDN_INLINE void executeKernel(const T0& D0, const T1& D1, const F& f)
    {
      sycl->queue.parallel_for(sycl::range<2>(D0, D1), [=](sycl::id<2> idx)
      {
        f(T0(idx[0]), T1(idx[1]));
      });
    }

    // Executes an ESIMD kernel on the device
    template <typename T0, typename T1, typename F>
    OIDN_INLINE void executeESIMDKernel(const T0& D0, const T1& D1, const F& f)
    {
      // FIXME: Named kernel is necessary due to an ESIMD bug
      sycl->queue.parallel_for<class ESIMDKernel>(sycl::range<2>(D0, D1), [=](sycl::id<2> idx) SYCL_ESIMD_KERNEL
      {
        f(T0(idx[0]), T1(idx[1]));
      });
    }

  protected:
    void init() override;
    void printInfo() override;
  };

} // namespace oidn
