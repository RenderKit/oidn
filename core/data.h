// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

OIDN_NAMESPACE_BEGIN

  // Opaque read-only data
  struct Data
  {
    const void* ptr;
    size_t size;

    Data() : ptr(nullptr), size(0) {}
    Data(std::nullptr_t) : ptr(nullptr), size(0) {}

    template<typename T>
    Data(T* ptr, size_t size)
      : ptr(ptr),
        size(size)
    {
      if (ptr == nullptr && size > 0)
        throw Exception(Error::InvalidArgument, "data pointer is null");
    }

    template<typename T, size_t N>
    Data(T (&array)[N]) : ptr(array), size(sizeof(array)) {}

    template<typename T, size_t N>
    Data& operator =(T (&array)[N])
    {
      ptr = array;
      size = sizeof(array);
      return *this;
    }

    oidn_inline operator bool() const
    {
      return ptr != nullptr;
    }
  };

OIDN_NAMESPACE_END
