// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include <CL/sycl.hpp>

namespace oidn {

  class SYCLDevice : public Device
  { 
  private:
    cl::sycl::device syclDevice;
    cl::sycl::context syclContext;
    cl::sycl::queue syclQueue;

  public:
    Ref<Buffer> newBuffer(size_t byteSize, Buffer::Kind kind) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;

    __forceinline cl::sycl::device&  getSYCLDevice()  { return syclDevice; }
    __forceinline cl::sycl::context& getSYCLContext() { return syclContext; }
    __forceinline cl::sycl::queue&   getSYCLQueue()   { return syclQueue; }

  protected:
    void init() override;
    void printInfo() override;
  };

} // namespace oidn
