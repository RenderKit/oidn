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

    Image(void* ptr, Format format, int width, int height, size_t byteOffset, size_t inByteItemStride, size_t inByteRowStride)
    {
      if (ptr == nullptr)
        throw Exception(Error::InvalidArgument, "buffer pointer null");

      init((char*)ptr + byteOffset, format, width, height, inByteItemStride, inByteRowStride);
    }

    Image(const Ref<Buffer>& buffer, Format format, int width, int height, size_t byteOffset, size_t inByteItemStride, size_t inByteRowStride)
    {
      init(buffer->data() + byteOffset, format, width, height, inByteItemStride, inByteRowStride);

      if (byteOffset + height * rowStride * byteItemStride > buffer->size())
        throw Exception(Error::InvalidArgument, "buffer region out of range");
    }

    void init(char* ptr, Format format, int width, int height, size_t inByteItemStride, size_t inByteRowStride)
    {
      assert(width >= 0);
      assert(height >= 0);
      if (width > maxSize || height > maxSize)
        throw Exception(Error::InvalidArgument, "image size too large");

      this->ptr = ptr;
      this->width = width;
      this->height = height;

      const size_t itemSize = getFormatBytes(format);
      if (inByteItemStride != 0)
      {
        if (inByteItemStride < itemSize)
          throw Exception(Error::InvalidArgument, "item stride smaller than item size");

        this->byteItemStride = inByteItemStride;
      }
      else
      {
        this->byteItemStride = itemSize;
      }

      if (inByteRowStride != 0)
      {
        if (inByteRowStride < width * this->byteItemStride)
          throw Exception(Error::InvalidArgument, "row stride smaller than width * item stride");
        if (inByteRowStride % this->byteItemStride != 0)
          throw Exception(Error::InvalidArgument, "row stride not integer multiple of item stride");

        this->rowStride = inByteRowStride / this->byteItemStride;
      }
      else
      {
        this->rowStride = width;
      }

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
