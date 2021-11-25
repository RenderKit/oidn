// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace oidn {

  // Opaque data
  struct Data
  {
    char* ptr;
    size_t size;

    Data() : ptr(nullptr), size(0) {}

    template<typename T>
    Data(T* ptr, size_t size) : ptr((char*)ptr), size(size) {}

    template<typename T, size_t N>
    Data(T (&array)[N]) : ptr((char*)array), size(sizeof(array)) {}

    template<typename T, size_t N>
    Data& operator =(T (&array)[N])
    {
      ptr = (char*)array;
      size = sizeof(array);
      return *this;
    }

    __forceinline operator bool() const
    {
      return ptr != nullptr;
    }
  };

} // namespace oidn
