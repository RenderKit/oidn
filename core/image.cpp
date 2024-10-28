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
      throw Exception(Error::InvalidArgument, "image size is too large");

    const size_t pixelByteSize = getFormatSize(format);
    if (pixelByteStride != 0)
    {
      if (pixelByteStride < pixelByteSize)
        throw Exception(Error::InvalidArgument, "pixel stride is smaller than pixel size");
      wByteStride = pixelByteStride;
    }
    else
      wByteStride = pixelByteSize;

    if (rowByteStride != 0)
    {
      if (rowByteStride < width * wByteStride)
        throw Exception(Error::InvalidArgument, "row stride is smaller than width * pixel stride");
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
      throw Exception(Error::InvalidArgument, "image pointer is null");

    this->ptr = static_cast<char*>(ptr) + byteOffset;
  }

  Image::Image(const Ref<Buffer>& buffer, const ImageDesc& desc, size_t byteOffset)
    : Memory(buffer, byteOffset),
      ImageDesc(desc)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");

    this->ptr = static_cast<char*>(buffer->getPtr()) + byteOffset;
  }

  Image::Image(const Ref<Buffer>& buffer, Format format, size_t width, size_t height, size_t byteOffset, size_t pixelByteStride, size_t rowByteStride)
    : Memory(buffer, byteOffset),
      ImageDesc(format, width, height, pixelByteStride, rowByteStride)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");

    this->ptr = static_cast<char*>(buffer->getPtr()) + byteOffset;
  }

  Image::Image(Engine* engine, Format format, size_t width, size_t height)
    : Memory(engine->newBuffer(width * height * getFormatSize(format), Storage::Device)),
      ImageDesc(format, width, height)
  {
    this->ptr = static_cast<char*>(buffer->getPtr());
  }

  void Image::postRealloc()
  {
    if (buffer)
      ptr = static_cast<char*>(buffer->getPtr()) + byteOffset;
  }

  bool Image::overlaps(const Image& other) const
  {
    if (!*this || !other)
      return false;

    // If any of the images are not backed by non-shared buffers, we cannot determine whether they
    // overlap due to potential virtual address aliasing, so we conservatively assume they do
    if (!buffer || buffer->isShared() || !other.buffer || other.buffer->isShared())
      return true;

    // If the images are backed by different non-shared buffers, they cannot overlap
    if (buffer != other.buffer)
      return false;

    // Check whether the memory ranges inside the same buffer overlap
    const char* begin1 = ptr;
    const char* end1   = ptr + getByteSize();
    const char* begin2 = other.ptr;
    const char* end2   = other.ptr + other.getByteSize();

    return begin1 < end2 && begin2 < end1;
  }

OIDN_NAMESPACE_END