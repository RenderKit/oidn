// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace oidn {

  class Device;

  // Generic buffer object
  class Buffer : public RefCount
  {
  public:
    enum class Kind
    {
      Host,
      Device,
      Shared,
      Unknown
    };

    virtual char* data() = 0;
    virtual const char* data() const = 0;
    virtual size_t size() const = 0;

    virtual void* map(size_t offset, size_t size) = 0;
    virtual void unmap(void* mappedPtr) = 0;

    // Resizes the buffer discarding its current contents
    virtual void resize(size_t newSize)
    {
      throw std::logic_error("resizing is not supported");
    }

    virtual Device* getDevice() = 0;
  };

  // Memory object backed by a buffer
  struct Memory
  {
    Ref<Buffer> buffer;  // buffer containing the data
    size_t bufferOffset; // offset in the buffer

    Memory() : bufferOffset(0) {}
    virtual ~Memory() = default;

    Memory(const Ref<Buffer>& buffer, size_t bufferOffset = 0)
      : buffer(buffer),
        bufferOffset(bufferOffset) {}

    // If the buffer gets reallocated, this must be called to update the internal pointer
    virtual void updatePtr() = 0;
  };

} // namespace oidn
