// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "buffer.h"

namespace oidn {

  struct ImageDesc
  {
    static constexpr int maxSize = 65536;

    int width;              // width in number of pixels
    int height;             // height in number of pixels
    size_t bytePixelStride; // pixel stride in number of *bytes*
    size_t rowStride;       // row stride in number of *pixel strides*
    Format format;          // pixel format

    __forceinline ImageDesc() = default;

    ImageDesc(Format format, size_t width, size_t height, size_t inBytePixelStride = 0, size_t inByteRowStride = 0)
    {
      if (width > maxSize || height > maxSize)
        throw Exception(Error::InvalidArgument, "image size too large");

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
        if (this->bytePixelStride != 0)
        {
          if (inByteRowStride < width * this->bytePixelStride)
            throw Exception(Error::InvalidArgument, "row stride smaller than width * pixel stride");
          if (inByteRowStride % this->bytePixelStride != 0)
            throw Exception(Error::InvalidArgument, "row stride not integer multiple of pixel stride");

          this->rowStride = inByteRowStride / this->bytePixelStride;
        }
        else
        {
          this->rowStride = inByteRowStride;
        }
      }
      else
      {
        this->rowStride = width;
      }

      this->format = format;
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

    // Returns the number of pixels in the image
    __forceinline size_t numElements() const
    {
      return size_t(width) * size_t(height);
    }

    // Returns the size in bytes of a pixel in the image
    __forceinline size_t elementByteSize() const
    {
      return getByteSize(format);
    }

    // Returns the size in bytes of the image
    __forceinline size_t byteSize() const
    {
      return size_t(height) * rowStride * bytePixelStride;
    }

    // Returns the aligned size in bytes of the image
    __forceinline size_t alignedByteSize() const
    {
      return round_up(byteSize(), memoryAlignment);
    }
  };

  class Image : public Memory, public ImageDesc
  {
  public:
    char* ptr; // pointer to the first pixel

    Image() :
      ImageDesc(Format::Undefined, 0, 0),
      ptr(nullptr) {}

    Image(void* ptr, Format format, size_t width, size_t height, size_t byteOffset, size_t inBytePixelStride, size_t inByteRowStride)
      : ImageDesc(format, width, height, inBytePixelStride, inByteRowStride)
    {
      if ((ptr == nullptr) && (byteOffset + byteSize() > 0))
        throw Exception(Error::InvalidArgument, "buffer region out of range");

      this->ptr = (char*)ptr + byteOffset;
    }

    Image(const Ref<Buffer>& buffer, const ImageDesc& desc, size_t byteOffset)
      : Memory(buffer, byteOffset),
        ImageDesc(desc)
    {
      if (byteOffset + byteSize() > buffer->size())
        throw Exception(Error::InvalidArgument, "buffer region out of range");

      this->ptr = buffer->data() + byteOffset;
    }

    Image(const Ref<Buffer>& buffer, Format format, size_t width, size_t height, size_t byteOffset, size_t inBytePixelStride, size_t inByteRowStride)
      : Memory(buffer, byteOffset),
        ImageDesc(format, width, height, inBytePixelStride, inByteRowStride)
    {
      if (byteOffset + byteSize() > buffer->size())
        throw Exception(Error::InvalidArgument, "buffer region out of range");

      this->ptr = buffer->data() + byteOffset;
    }

    Image(const Ref<Device>& device, Format format, size_t width, size_t height)
      : Memory(device->newBuffer(width * height * getByteSize(format))),
        ImageDesc(format, width, height)
    {
      this->ptr = buffer->data();
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

    __forceinline const ImageDesc& desc() const { return *this; }

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

    void updatePtr() override
    {
      if (buffer)
      {
        if (bufferOffset + byteSize() > buffer->size())
          throw Exception(Error::Unknown, "buffer region out of range");

        ptr = buffer->data() + bufferOffset;
      }
    }
  };

} // namespace oidn
