// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

namespace oidn {

  template<typename T>
  struct ImageAccessor
  {
    uint8_t* ptr;
    size_t wByteStride; // pixel stride in number of bytes
    size_t hByteStride; // row stride in number of bytes
    int W, H;           // width, height

    OIDN_HOST_DEVICE_INLINE size_t getByteOffset(int h, int w) const
    {
      return (size_t)h * hByteStride + (size_t)w * wByteStride;
    }

    OIDN_HOST_DEVICE_INLINE vec3<T> get3(int h, int w) const
    {
      const size_t byteOffset = getByteOffset(h, w);
      T* pixel = (T*)&ptr[byteOffset];
      return vec3<T>(pixel[0], pixel[1], pixel[2]);
    }

    OIDN_HOST_DEVICE_INLINE void set3(int h, int w, const vec3<T>& value) const
    {
      const size_t byteOffset = getByteOffset(h, w);
      T* pixel = (T*)&ptr[byteOffset];
      pixel[0] = value.x;
      pixel[1] = value.y;
      pixel[2] = value.z;
    }
  };

} // namespace oidn
