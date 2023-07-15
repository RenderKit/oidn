// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  ImageDesc::ImageDesc(Format format, size_t width, size_t height, size_t pixelByteStride, size_t rowByteStride)
    : width(width),
      height(height),
      format(format)
  {
    if (width > maxDim || height > maxDim || width * height * getC() > std::numeric_limits<int>::max())
      throw Exception(Error::InvalidArgument, "image size too large");

    const size_t pixelByteSize = getFormatSize(format);
    if (pixelByteStride != 0)
    {
      if (pixelByteStride < pixelByteSize)
        throw Exception(Error::InvalidArgument, "pixel stride smaller than pixel size");
      wByteStride = pixelByteStride;
    }
    else
      wByteStride = pixelByteSize;

    if (rowByteStride != 0)
    {
      if (rowByteStride < width * wByteStride)
        throw Exception(Error::InvalidArgument, "row stride smaller than width * pixel stride");
      hByteStride = rowByteStride;
    }
    else
      hByteStride = width * wByteStride;
  }

  Image::Image() :
    ImageDesc(Format::Undefined, 0, 0),
    ptr(nullptr) {}

  Image::Image(void* ptr, Format format, size_t width, size_t height, size_t byteOffset, size_t pixelByteStride, size_t rowByteStride)
    : ImageDesc(format, width, height, pixelByteStride, rowByteStride)
  {
    if ((ptr == nullptr) && (byteOffset + getByteSize() > 0))
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    this->ptr = static_cast<char*>(ptr) + byteOffset;
  }

  Image::Image(const Ref<Buffer>& buffer, const ImageDesc& desc, size_t byteOffset)
    : Memory(buffer, byteOffset),
      ImageDesc(desc)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    this->ptr = buffer->getPtr() + byteOffset;
  }

  Image::Image(const Ref<Buffer>& buffer, Format format, size_t width, size_t height, size_t byteOffset, size_t pixelByteStride, size_t rowByteStride)
    : Memory(buffer, byteOffset),
      ImageDesc(format, width, height, pixelByteStride, rowByteStride)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    this->ptr = buffer->getPtr() + byteOffset;
  }

  Image::Image(const Ref<Engine>& engine, Format format, size_t width, size_t height)
    : Memory(engine->newBuffer(width * height * getFormatSize(format), Storage::Device)),
      ImageDesc(format, width, height)
  {
    this->ptr = buffer->getPtr();
  }

  void Image::updatePtr()
  {
    if (buffer)
    {
      if (byteOffset + getByteSize() > buffer->getByteSize())
        throw std::range_error("buffer region out of range");

      ptr = buffer->getPtr() + byteOffset;
    }
  }

  bool Image::overlaps(const Image& other) const
  {
    if (!*this || !other)
      return false;

    // If the images are backed by different buffers, they cannot overlap
    if (buffer != other.buffer)
      return false;

    // Check whether the pointer intervals overlap
    const char* begin1 = ptr;
    const char* end1   = ptr + getByteSize();
    const char* begin2 = other.ptr;
    const char* end2   = other.ptr + other.getByteSize();

    return begin1 < end2 && begin2 < end1;
  }

OIDN_NAMESPACE_END