// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.h"
#include "exception.h"

OIDN_NAMESPACE_BEGIN

  // Plain value or structure stored in a buffer
  template<typename T>
  class Record final : public Memory
  {
    static_assert(std::is_trivial<T>::value, "record can be used only for trivial types");

  public:
    Record(const Ref<Buffer>& buffer, size_t byteOffset = 0)
      : Memory(buffer, byteOffset)
    {
      if (byteOffset + sizeof(T) > buffer->getByteSize())
        throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    }

    T* getPtr() const
    {
      return (T*)((char*)buffer->getPtr() + byteOffset);
    }
  };

OIDN_NAMESPACE_END