// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"
#include <vector>

namespace oidn {

  // Generic tensor
  class Tensor
  {
  public:
    using Dims = std::vector<int64_t>;

    float* data;
    Dims dims;
    std::string layout;
    std::shared_ptr<std::vector<char>> buffer; // optional, only for reference counting

    __forceinline Tensor() : data(nullptr) {}

    __forceinline Tensor(const Dims& dims, const std::string& layout)
      : dims(dims),
        layout(layout)
    {
      buffer = std::make_shared<std::vector<char>>(size() * sizeof(float));
      data = (float*)buffer->data();
    }

    __forceinline Tensor(const Dims& dims, const std::string& layout, float* data)
      : data(data),
        dims(dims),
        layout(layout)
    {}

    __forceinline operator bool() const { return data != nullptr; }

    __forceinline int ndims() const { return (int)dims.size(); }

    // Returns the number of values
    __forceinline size_t size() const
    {
      size_t size = 1;
      for (int i = 0; i < ndims(); ++i)
        size *= dims[i];
      return size;
    }

    // Returns the size in bytes
    __forceinline size_t byteSize() const
    {
      return size() * sizeof(float);
    }

    __forceinline float& operator [](size_t i) { return data[i]; }
    __forceinline const float& operator [](size_t i) const { return data[i]; }

    __forceinline float& operator ()(int64_t i0) { return data[getIndex(i0)]; }
    __forceinline const float& operator ()(int64_t i0) const { return data[getIndex(i0)]; }

    __forceinline float& operator ()(int64_t i0, int64_t i1)
    { return data[getIndex(i0, i1)]; }
    __forceinline const float& operator ()(int64_t i0, int64_t i1) const
    { return data[getIndex(i0, i1)]; }

    __forceinline float& operator ()(int64_t i0, int64_t i1, int64_t i2)
    { return data[getIndex(i0, i1, i2)]; }
    __forceinline const float& operator ()(int64_t i0, int64_t i1, int64_t i2) const
    { return data[getIndex(i0, i1, i2)]; }

    __forceinline float& operator ()(int64_t i0, int64_t i1, int64_t i2, int64_t i3)
    { return data[getIndex(i0, i1, i2, i3)]; }
    __forceinline const float& operator ()(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const
    { return data[getIndex(i0, i1, i2, i3)]; }

  private:
    __forceinline int64_t getIndex(int64_t i0) const
    {
      assert(ndims() == 1);
      assert(i0 < dims[0]);
      return i0;
    }

    __forceinline int64_t getIndex(int64_t i0, int64_t i1) const
    {
      assert(ndims() == 2);
      assert(i0 < dims[0] && i1 < dims[1]);
      return i0 * dims[1] + i1;
    }

    __forceinline int64_t getIndex(int64_t i0, int64_t i1, int64_t i2) const
    {
      assert(ndims() == 3);
      assert(i0 < dims[0] && i1 < dims[1] && i2 < dims[2]);
      return (i0 * dims[1] + i1) * dims[2] + i2;
    }

    __forceinline int64_t getIndex(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const
    {
      assert(ndims() == 4);
      assert(i0 < dims[0] && i1 < dims[1] && i2 < dims[2] && i3 < dims[3]);
      return ((i0 * dims[1] + i1) * dims[2] + i2) * dims[3] + i3;
    }
  };

} // namespace oidn
