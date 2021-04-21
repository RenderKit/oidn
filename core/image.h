// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "buffer.h"

namespace oidn {

  struct Image
  {
    static constexpr int maxSize = 65536;

    char* ptr;              // pointer to the first pixel
    int width;              // width in number of pixels
    int height;             // height in number of pixels
    size_t bytePixelStride; // pixel stride in number of *bytes*
    size_t rowStride;       // row stride in number of *pixel strides*
    Format format;          // pixel format
    Ref<Buffer> buffer;     // buffer containing the image data

    Image() : ptr(nullptr), width(0), height(0), bytePixelStride(0), rowStride(0), format(Format::Undefined) {}

    Image(void* ptr, Format format, size_t width, size_t height, size_t byteOffset, size_t inBytePixelStride, size_t inByteRowStride)
    {
      if (ptr == nullptr)
        throw Exception(Error::InvalidArgument, "buffer pointer null");

      init((char*)ptr + byteOffset, format, width, height, inBytePixelStride, inByteRowStride);
    }

    Image(const Ref<Buffer>& buffer, Format format, size_t width, size_t height, size_t byteOffset, size_t inBytePixelStride, size_t inByteRowStride)
      : buffer(buffer)
    {
      init(buffer->data() + byteOffset, format, width, height, inBytePixelStride, inByteRowStride);

      if (byteOffset + height * rowStride * bytePixelStride > buffer->size())
        throw Exception(Error::InvalidArgument, "buffer region out of range");
    }

    Image(const Ref<Device>& device, Format format, size_t width, size_t height)
      : buffer(makeRef<Buffer>(device, width * height * getByteSize(format)))
    {
      init(buffer->data(), format, width, height);
    }

    void init(char* ptr, Format format, size_t width, size_t height, size_t inBytePixelStride = 0, size_t inByteRowStride = 0)
    {
      if (width > maxSize || height > maxSize)
        throw Exception(Error::InvalidArgument, "image size too large");

      this->ptr = ptr;
      this->width  = int(width);
      this->height = int(height);

      const size_t pixelSize = getByteSize(format);
      if (inBytePixelStride != 0)
      {
        if (inBytePixelStride < pixelSize)
          throw Exception(Error::InvalidArgument, "pixel stride smaller than pixel size");

        this->bytePixelStride = inBytePixelStride;
      }
      else
      {
        this->bytePixelStride = pixelSize;
      }

      if (inByteRowStride != 0)
      {
        if (inByteRowStride < width * this->bytePixelStride)
          throw Exception(Error::InvalidArgument, "row stride smaller than width * pixel stride");
        if (inByteRowStride % this->bytePixelStride != 0)
          throw Exception(Error::InvalidArgument, "row stride not integer multiple of pixel stride");

        this->rowStride = inByteRowStride / this->bytePixelStride;
      }
      else
      {
        this->rowStride = width;
      }

      this->format = format;
    }

    __forceinline char* get(int h, int w)
    {
      return ptr + ((size_t(h) * rowStride + size_t(w)) * bytePixelStride);
    }

    __forceinline const char* get(int h, int w) const
    {
      return ptr + ((size_t(h) * rowStride + size_t(w)) * bytePixelStride);
    }

    __forceinline       char* begin()       { return ptr; }
    __forceinline const char* begin() const { return ptr; }

    __forceinline       char* end()       { return get(height, 0); }
    __forceinline const char* end() const { return get(height, 0); }

    __forceinline operator bool() const
    {
      return ptr != nullptr;
    }

    // Returns the number of channels
    __forceinline int numChannels() const
    {
      switch (format)
      {
      case Format::Undefined:
        return 0;
      case Format::Float:
        return 1;
      case Format::Float2:
        return 2;
      case Format::Float3:
        return 3;
      case Format::Float4:
        return 4;
      default:
        throw Exception(Error::InvalidArgument, "invalid image format");
      }
    }

    // Determines whether two images overlap in memory
    bool overlaps(const Image& other) const
    {
      if (!ptr || !other.ptr)
        return false;

      // If the images are backed by different buffers, they cannot overlap
      if (buffer != other.buffer)
        return false;

      // Check whether the pointer intervals overlap
      const char* begin1 = begin();
      const char* end1   = end();
      const char* begin2 = other.begin();
      const char* end2   = other.end();

      return begin1 < end2 && begin2 < end1;
    }

    // Converts to ISPC equivalent
    operator ispc::Image() const
    {
      ispc::Image result;
      result.ptr = (uint8_t*)ptr;
      result.rowStride = rowStride;
      result.bytePixelStride = bytePixelStride;
      return result;
    }
  };

} // namespace oidn
