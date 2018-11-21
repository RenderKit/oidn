// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

#include "common.h"

namespace oidn {

  inline size_t getDataStride(Format format)
  {
    switch (format)
    {
    case Format::Undefined: return 1;
    case Format::Float:     return sizeof(float);
    case Format::Float2:    return sizeof(float)*2;
    case Format::Float3:    return sizeof(float)*3;
    }
    assert(0);
    return 0;
  }

  struct Data2D
  {
    char* ptr;
    int width;
    int height;
    size_t stride;
    size_t rowStride;
    Format format;
    Ref<Buffer> buffer;

    Data2D() : ptr(nullptr), width(0), height(0), stride(0), rowStride(0), format(Format::Undefined) {}

    Data2D(void* ptr, Format format, int width, int height, size_t offset, size_t stride, size_t rowStride)
    {
      init((char*)ptr + offset, format, width, height, stride, rowStride);
    }

    Data2D(const Ref<Buffer>& buffer, Format format, int width, int height, size_t offset, size_t stride, size_t rowStride)
    {
      init(buffer->data() + offset, format, width, height, stride, rowStride);
    }

    void init(char* ptr, Format format, int width, int height, size_t stride, size_t rowStride)
    {
      this->ptr = ptr;
      this->width = width;
      this->height = height;
      this->stride = (stride != 0) ? stride : getDataStride(format);
      this->rowStride = (rowStride != 0) ? rowStride : (width * this->stride);
      this->format = format;
    }

    __forceinline char* get(int y, int x)
    {
      return ptr + (size_t(y) * rowStride) + (size_t(x) * stride);
    }

    __forceinline const char* get(int y, int x) const
    {
      return ptr + (size_t(y) * rowStride) + (size_t(x) * stride);
    }

    operator bool() const
    {
      return ptr != nullptr;
    }
  };

} // ::oidn
