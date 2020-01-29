// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "platform.h"
#include <vector>
#include <map>

namespace oidn {

  // Generic tensor
  class Tensor
  {
  public:
    using Dims = std::vector<int64_t>;

    float* data;
    Dims dims;
    std::string format;
    std::shared_ptr<std::vector<char>> buffer; // optional, only for reference counting

    __forceinline Tensor() : data(nullptr) {}

    __forceinline Tensor(const Dims& dims, const std::string& format)
      : dims(dims),
        format(format)
    {
      buffer = std::make_shared<std::vector<char>>(size() * sizeof(float));
      data = (float*)buffer->data();
    }

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

    __forceinline float& operator [](size_t i) { return data[i]; }
    __forceinline const float& operator [](size_t i) const { return data[i]; }

    __forceinline float& operator ()(size_t i0) { return data[getIndex(i0)]; }
    __forceinline const float& operator ()(size_t i0) const { return data[getIndex(i0)]; }

    __forceinline float& operator ()(size_t i0, size_t i1)
    { return data[getIndex(i0, i1)]; }
    __forceinline const float& operator ()(size_t i0, size_t i1) const
    { return data[getIndex(i0, i1)]; }

    __forceinline float& operator ()(size_t i0, size_t i1, size_t i2)
    { return data[getIndex(i0, i1, i2)]; }
    __forceinline const float& operator ()(size_t i0, size_t i1, size_t i2) const
    { return data[getIndex(i0, i1, i2)]; }

    __forceinline float& operator ()(size_t i0, size_t i1, size_t i2, size_t i3)
    { return data[getIndex(i0, i1, i2, i3)]; }
    __forceinline const float& operator ()(size_t i0, size_t i1, size_t i2, size_t i3) const
    { return data[getIndex(i0, i1, i2, i3)]; }

  private:
    __forceinline size_t getIndex(size_t i0) const
    {
      assert(ndims() == 1);
      assert(i0 < dims[0]);
      return i0;
    }

    __forceinline size_t getIndex(size_t i0, size_t i1) const
    {
      assert(ndims() == 2);
      assert(i0 < dims[0] && i1 < dims[1]);
      return i0 * dims[1] + i1;
    }

    __forceinline size_t getIndex(size_t i0, size_t i1, size_t i2) const
    {
      assert(ndims() == 3);
      assert(i0 < dims[0] && i1 < dims[1] && i2 < dims[2]);
      return (i0 * dims[1] + i1) * dims[2] + i2;
    }

    __forceinline size_t getIndex(size_t i0, size_t i1, size_t i2, size_t i3) const
    {
      assert(ndims() == 4);
      assert(i0 < dims[0] && i1 < dims[1] && i2 < dims[2] && i3 < dims[3]);
      return ((i0 * dims[1] + i1) * dims[2] + i2) * dims[3] + i3;
    }
  };

  // Parses tensors from a buffer
  std::map<std::string, Tensor> parseTensors(void* buffer);

} // namespace oidn
