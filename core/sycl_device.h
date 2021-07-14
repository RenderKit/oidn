// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"

namespace oidn {

  class SYCLDevice : public Device
  { 
  private:
    sycl::device syclDevice;
    sycl::context syclContext;
    sycl::queue syclQueue;

  public:
    Ref<Buffer> newBuffer(size_t byteSize, Buffer::Kind kind) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;

    __forceinline sycl::device&  getSYCLDevice()  { return syclDevice; }
    __forceinline sycl::context& getSYCLContext() { return syclContext; }
    __forceinline sycl::queue&   getSYCLQueue()   { return syclQueue; }

  protected:
    void init() override;
    void printInfo() override;
  };

} // namespace oidn
