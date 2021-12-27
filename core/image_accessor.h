// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

namespace oidn {

  template<typename T>
  struct ImageAccessor
  {
    uint8_t* ptr;
    size_t bytePixelStride; // pixel stride in number of *bytes*
    size_t rowStride;       // row stride in number of *pixel strides*

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int h, int w) const
    {
      return (((size_t)h * rowStride + (size_t)w) * bytePixelStride);
    }

    OIDN_HOST_DEVICE_INLINE vec3<T> get3(int h, int w) const
    {
      const size_t offset = getOffset(h, w);
      T* pixel = (T*)&ptr[offset];
      return vec3<T>(pixel[0], pixel[1], pixel[2]);
    }

    OIDN_HOST_DEVICE_INLINE void set3(int h, int w, const vec3<T>& value) const
    {
      const size_t offset = getOffset(h, w);
      T* pixel = (T*)&ptr[offset];
      pixel[0] = value.x;
      pixel[1] = value.y;
      pixel[2] = value.z;
    }
  };

} // namespace oidn
