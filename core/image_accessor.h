// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include "vec.h"

OIDN_NAMESPACE_BEGIN

  struct ImageAccessor
  {
    char* ptr;
    size_t hByteStride; // row stride in number of bytes
    size_t wByteStride; // pixel stride in number of bytes
    DataType dataType;  // data type
    int C, H, W;        // channels (1-3), height, width

    OIDN_HOST_DEVICE_INLINE size_t getByteOffset(int h, int w) const
    {
      return size_t(h) * hByteStride + size_t(w) * wByteStride;
    }

    template<typename T = float>
    OIDN_HOST_DEVICE_INLINE vec3<T> get3(int h, int w) const
    {
      const void* pixelPtr = ptr + getByteOffset(h, w);
      if (dataType == DataType::Float32)
      {
        const float* pixel = static_cast<const float*>(pixelPtr);
        if (C == 3)
          return vec3<T>(pixel[0], pixel[1], pixel[2]);
        else if (C == 2)
          return vec3<T>(pixel[0], pixel[1], pixel[1]);
        else // if (C == 1)
          return vec3<T>(pixel[0], pixel[0], pixel[0]);
      }
      else // if (dataType == DataType::Float16)
      {
        const half* pixel = static_cast<const half*>(pixelPtr);
        if (C == 3)
          return vec3<T>(pixel[0], pixel[1], pixel[2]);
        else if (C == 2)
          return vec3<T>(pixel[0], pixel[1], pixel[1]);
        else // if (C == 1)
          return vec3<T>(pixel[0], pixel[0], pixel[0]);
      }
    }

    template<typename T>
    OIDN_HOST_DEVICE_INLINE void set3(int h, int w, const vec3<T>& value) const
    {
      void* pixelPtr = ptr + getByteOffset(h, w);
      if (dataType == DataType::Float32)
      {
        float* pixel = static_cast<float*>(pixelPtr);
        if (C == 3)
        {
          pixel[0] = value.x;
          pixel[1] = value.y;
          pixel[2] = value.z;
        }
        else if (C == 2)
        {
          pixel[0] = value.x;
          pixel[1] = value.y;
        }
        else // if (C == 1)
          pixel[0] = value.x;
      }
      else // if (dataType == DataType::Float16)
      {
        half* pixel = static_cast<half*>(pixelPtr);
        if (C == 3)
        {
          pixel[0] = value.x;
          pixel[1] = value.y;
          pixel[2] = value.z;
        }
        else if (C == 2)
        {
          pixel[0] = value.x;
          pixel[1] = value.y;
        }
        else // if (C == 1)
          pixel[0] = value.x;
      }
    }
  };

OIDN_NAMESPACE_END
