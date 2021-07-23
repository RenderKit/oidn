// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"

namespace oidn {

  class SYCLDevice : public Device
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

    __forceinline sycl::device&  getSYCLDevice()  { return sycl->device; }
    __forceinline sycl::context& getSYCLContext() { return sycl->context; }
    __forceinline sycl::queue&   getSYCLQueue()   { return sycl->queue; }

  protected:
    void init() override;
    void printInfo() override;
  };

} // namespace oidn
