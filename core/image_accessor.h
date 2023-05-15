// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include "vec.h"

OIDN_NAMESPACE_BEGIN

  template<typename T>
  struct ImageAccessor
  {
    char* ptr;
    size_t wByteStride; // pixel stride in number of bytes
    size_t hByteStride; // row stride in number of bytes
    int W, H;           // width, height

    OIDN_HOST_DEVICE_INLINE size_t getByteOffset(int h, int w) const
    {
      return size_t(h) * hByteStride + size_t(w) * wByteStride;
    }

    OIDN_HOST_DEVICE_INLINE vec3<T> get3(int h, int w) const
    {
      T* pixel = reinterpret_cast<T*>(ptr + getByteOffset(h, w));
      return vec3<T>(pixel[0], pixel[1], pixel[2]);
    }

    OIDN_HOST_DEVICE_INLINE void set3(int h, int w, const vec3<T>& value) const
    {
      T* pixel = reinterpret_cast<T*>(ptr + getByteOffset(h, w));
      pixel[0] = value.x;
      pixel[1] = value.y;
      pixel[2] = value.z;
    }
  };

OIDN_NAMESPACE_END
