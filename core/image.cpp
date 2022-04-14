// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image.h"

namespace oidn {

  ImageDesc::ImageDesc(Format format, size_t width, size_t height, size_t bytePixelStride, size_t byteRowStride)
    : width(width),
      height(height),
      format(format)
  {
    if (width > maxDim || height > maxDim || width * height * getC() > std::numeric_limits<int>::max())
      throw Exception(Error::InvalidArgument, "image size too large");

    const size_t pixelSize = getFormatSize(format);
    if (bytePixelStride != 0)
    {
      if (bytePixelStride < pixelSize)
        throw Exception(Error::InvalidArgument, "pixel stride smaller than pixel size");
      wStride = bytePixelStride;
    }
    else
      wStride = pixelSize;

    if (byteRowStride != 0)
    {
      if (byteRowStride < width * wStride)
        throw Exception(Error::InvalidArgument, "row stride smaller than width * pixel stride");
      hStride = byteRowStride;
    }
    else
      hStride = width * wStride;
  }

  Image::Image() :
    ImageDesc(Format::Undefined, 0, 0),
    ptr(nullptr) {}

  Image::Image(void* ptr, Format format, size_t width, size_t height, size_t byteOffset, size_t bytePixelStride, size_t byteRowStride)
    : ImageDesc(format, width, height, bytePixelStride, byteRowStride)
  {
    if ((ptr == nullptr) && (byteOffset + getByteSize() > 0))
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    this->ptr = (char*)ptr + byteOffset;
  }

  Image::Image(const Ref<Buffer>& buffer, const ImageDesc& desc, size_t byteOffset)
    : Memory(buffer, byteOffset),
      ImageDesc(desc)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    this->ptr = buffer->getData() + byteOffset;
  }

  Image::Image(const Ref<Buffer>& buffer, Format format, size_t width, size_t height, size_t byteOffset, size_t bytePixelStride, size_t byteRowStride)
    : Memory(buffer, byteOffset),
      ImageDesc(format, width, height, bytePixelStride, byteRowStride)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    this->ptr = buffer->getData() + byteOffset;
  }

  Image::Image(const Ref<Device>& device, Format format, size_t width, size_t height)
    : Memory(device->newBuffer(width * height * getFormatSize(format), Storage::Device)),
      ImageDesc(format, width, height)
  {
    this->ptr = buffer->getData();
  }

  Image::operator ispc::ImageAccessor() const
  {
    ispc::ImageAccessor acc;
    acc.ptr = (uint8_t*)ptr;
    acc.hStride = hStride;
    acc.wStride = wStride;

    if (format != Format::Undefined)
    {
      switch (getDataType())
      {
      case DataType::Float32:
        acc.dataType = ispc::DataType_Float32;
        break;
      case DataType::Float16:
        acc.dataType = ispc::DataType_Float16;
        break;
      case DataType::UInt8:
        acc.dataType = ispc::DataType_UInt8;
        break;
      default:
        throw std::logic_error("unsupported data type");
      }
    }
    else
      acc.dataType = ispc::DataType_Float32;

    acc.W = width;
    acc.H = height;

    return acc;
  }

  void Image::updatePtr()
  {
    if (buffer)
    {
      if (byteOffset + getByteSize() > buffer->getByteSize())
        throw std::range_error("buffer region out of range");

      ptr = buffer->getData() + byteOffset;
    }
  }

  bool Image::overlaps(const Image& other) const
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

} // namespace oidn