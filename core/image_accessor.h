// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

namespace oidn {

  template<typename T>
  struct ImageAccessor
  {
    uint8_t* ptr;
    size_t wStride; // pixel stride in number of bytes
    size_t hStride; // row stride in number of bytes
    int W, H;       // width, height

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int h, int w) const
    {
      return (size_t)h * hStride + (size_t)w * wStride;
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
