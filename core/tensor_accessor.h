// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_layout.h"
#include "vec.h"

OIDN_NAMESPACE_BEGIN

  template<typename T>
  struct TensorAccessor1D
  {
    TensorByteOffset<T, TensorLayout::x> getByteOffset;
    oidn_global char* ptr;
    int X; // padded dimensions

    TensorAccessor1D() = default;

    OIDN_HOST_DEVICE_INLINE TensorAccessor1D(oidn_global void* data, int X)
      : ptr(static_cast<oidn_global char*>(data)),
        X(X) {}

    OIDN_HOST_DEVICE_INLINE oidn_global T& operator ()(int x) const
    {
      return *reinterpret_cast<oidn_global T*>(ptr + getByteOffset(x));
    }
  };

  template<typename T, TensorLayout layout>
  struct TensorAccessor3D
  {
    TensorByteOffset<T, layout> getByteOffset;
    oidn_global char* ptr;
    int C, H, W; // padded dimensions

    TensorAccessor3D() = default;

    OIDN_HOST_DEVICE_INLINE TensorAccessor3D(oidn_global void* data, int C, int H, int W)
      : getByteOffset(C, H, W),
        ptr(static_cast<oidn_global char*>(data)), C(C), H(H), W(W) {}

    OIDN_HOST_DEVICE_INLINE oidn_global T& operator ()(int c, int h, int w) const
    {
      return *reinterpret_cast<oidn_global T*>(ptr + getByteOffset(c, h, w));
    }

    OIDN_HOST_DEVICE_INLINE vec3<T> get3(int c, int h, int w) const
    {
      return vec3<T>((*this)(c,   h, w),
                     (*this)(c+1, h, w),
                     (*this)(c+2, h, w));
    }

    OIDN_HOST_DEVICE_INLINE void set3(int c, int h, int w, vec3<T> value) const
    {
      (*this)(c,   h, w) = value.x;
      (*this)(c+1, h, w) = value.y;
      (*this)(c+2, h, w) = value.z;
    }
  };

  template<typename T, TensorLayout layout>
  struct TensorAccessor4D
  {
    TensorByteOffset<T, layout> getByteOffset;
    oidn_global char* ptr;
    int O, I, H, W; // padded dimensions

    TensorAccessor4D() = default;

    OIDN_HOST_DEVICE_INLINE TensorAccessor4D(oidn_global void* data, int O, int I, int H, int W)
      : getByteOffset(O, I, H, W),
        ptr(static_cast<oidn_global char*>(data)), O(O), I(I), H(H), W(W) {}

    OIDN_HOST_DEVICE_INLINE oidn_global T& operator ()(int o, int i, int h, int w) const
    {
      return *reinterpret_cast<oidn_global T*>(ptr + getByteOffset(o, i, h, w));
    }
  };

OIDN_NAMESPACE_END
