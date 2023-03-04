// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_layout.h"
#include "vec.h"

OIDN_NAMESPACE_BEGIN

  template<typename T>
  struct TensorAccessor1D
  {
    T* ptr;
    int X; // padded dimensions

    TensorAccessor1D() = default;

    OIDN_HOST_DEVICE_INLINE TensorAccessor1D(const void* data, int X)
      : ptr((T*)data), X(X) {}

    OIDN_HOST_DEVICE_INLINE T& operator ()(int x) const
    {
      return ptr[x];
    }
  };

  template<typename T, TensorLayout layout>
  struct TensorAccessor3D : TensorAddressing<T, layout>
  {
    char* ptr;
    int C, H, W; // padded dimensions

    TensorAccessor3D() = default;

    OIDN_HOST_DEVICE_INLINE TensorAccessor3D(const void* data, int C, int H, int W)
      : TensorAddressing<T, layout>(C, H, W),
        ptr((char*)data), C(C), H(H), W(W) {}

    OIDN_HOST_DEVICE_INLINE T& operator ()(int c, int h, int w) const
    {
      return *(T*)(ptr + this->getByteOffset(c, h, w));
    }
    
    OIDN_HOST_DEVICE_INLINE vec3<T> get3(int c, int h, int w) const
    {
      return vec3<T>((*this)(c,   h, w),
                     (*this)(c+1, h, w),
                     (*this)(c+2, h, w));
    }

    OIDN_HOST_DEVICE_INLINE void set3(int c, int h, int w, const vec3<T>& value) const
    {
      (*this)(c,   h, w) = value.x;
      (*this)(c+1, h, w) = value.y;
      (*this)(c+2, h, w) = value.z;
    }
  };

  template<typename T, TensorLayout layout>
  struct TensorAccessor4D : TensorAddressing<T, layout>
  {
    char* ptr;
    int O, I, H, W; // padded dimensions

    TensorAccessor4D() = default;

    OIDN_HOST_DEVICE_INLINE TensorAccessor4D(const void* data, int O, int I, int H, int W)
      : TensorAddressing<T, layout>(O, I, H, W),
        ptr((char*)data), O(O), I(I), H(H), W(W) {}

    OIDN_HOST_DEVICE_INLINE T& operator ()(int o, int i, int h, int w) const
    {
      return *(T*)(ptr + this->getByteOffset(o, i, h, w));
    }
  };

OIDN_NAMESPACE_END
