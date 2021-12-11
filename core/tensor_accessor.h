// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor_layout.h"

namespace oidn {

  template<typename T>
  struct TensorAccessor1D
  {
    T* ptr;
    int X;

    TensorAccessor1D() {}

    TensorAccessor1D(const void* data, int X)
      : ptr((T*)data), X(X) {}

    __forceinline T& operator ()(int x) const
    {
      return ptr[x];
    }
  };

  template<typename T, TensorLayout layout>
  struct TensorAccessor3D : TensorAddressing<T, layout>
  {
    char* ptr;
    int C, H, W;

    TensorAccessor3D() {}

    TensorAccessor3D(const void* data, int C, int H, int W)
      : TensorAddressing<T, layout>(C, H, W),
        ptr((char*)data), C(C), H(H), W(W) {}

    __forceinline T& operator ()(int c, int h, int w) const
    {
      return *(T*)(ptr + this->getOffset(c, h, w));
    }
    
    __forceinline vec3<T> get3(int c, int h, int w) const
    {
      return vec3<T>((*this)(c,   h, w),
                     (*this)(c+1, h, w),
                     (*this)(c+2, h, w));
    }

    __forceinline void set3(int c, int h, int w, const vec3<T>& value) const
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
    int O, I, H, W;

    TensorAccessor4D() {}

    TensorAccessor4D(const void* data, int O, int I, int H, int W)
      : TensorAddressing<T, layout>(O, I, H, W),
        ptr((char*)data), O(O), I(I), H(H), W(W) {}

    __forceinline T& operator ()(int o, int i, int h, int w) const
    {
      return *(T*)(ptr + this->getOffset(o, i, h, w));
    }
  };

} // namespace oidn
