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
#include "buffer.h"

namespace oidn {

  struct Image
  {
    static constexpr int maxSize = 65536;

    char* ptr;             // pointer to the first item
    int width;             // width in number of items
    int height;            // height in number of items
    size_t byteItemStride; // item stride in number of *bytes*
    size_t rowStride;      // row stride in number of *item strides*
    Format format;         // item format
    Ref<Buffer> buffer;    // buffer containing the image data

    Image() : ptr(nullptr), width(0), height(0), byteItemStride(0), rowStride(0), format(Format::Undefined) {}

    Image(void* ptr, Format format, int width, int height, size_t byteOffset, size_t byteItemStride, size_t byteRowStride)
    {
      if (ptr == nullptr)
        throw Exception(Error::InvalidArgument, "buffer pointer cannot be null");
      init((char*)ptr + byteOffset, format, width, height, byteItemStride, byteRowStride);
    }

    Image(const Ref<Buffer>& buffer, Format format, int width, int height, size_t byteOffset, size_t byteItemStride, size_t byteRowStride)
    {
      init(buffer->data() + byteOffset, format, width, height, byteItemStride, byteRowStride);
    }

    void init(char* ptr, Format format, int width, int height, size_t byteItemStride, size_t byteRowStride)
    {
      assert(width >= 0);
      assert(height >= 0);
      if (width > maxSize || height > maxSize)
        throw Exception(Error::InvalidArgument, "image size is too large");

      this->ptr = ptr;
      this->width = width;
      this->height = height;

      this->byteItemStride = (byteItemStride != 0) ? byteItemStride : getFormatBytes(format);
      if (byteRowStride != 0)
      {
        if (byteRowStride % this->byteItemStride != 0)
          throw Exception(Error::InvalidArgument, "the row stride must be a multiple of the item stride");
        this->rowStride = byteRowStride / this->byteItemStride;
      }
      else
        this->rowStride = width;

      this->format = format;
    }

    __forceinline char* get(int y, int x)
    {
      return ptr + ((size_t(y) * rowStride + size_t(x)) * byteItemStride);
    }

    __forceinline const char* get(int y, int x) const
    {
      return ptr + ((size_t(y) * rowStride + size_t(x)) * byteItemStride);
    }

    operator bool() const
    {
      return ptr != nullptr;
    }
  };

} // namespace oidn
