// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.h"
#include "image_accessor.h"
#include "exception.h"

OIDN_NAMESPACE_BEGIN

  class Engine;

  struct ImageDesc
  {
    static constexpr size_t maxDim = 65536;

    size_t width;       // width in number of pixels
    size_t height;      // height in number of pixels
    size_t wByteStride; // pixel stride in number of bytes
    size_t hByteStride; // row stride in number of bytes
    Format format;      // pixel format

    ImageDesc() = default;
    ImageDesc(Format format, size_t width, size_t height, size_t pixelByteStride = 0, size_t rowByteStride = 0);

    // Returns the number of channels
    oidn_inline int getC() const
    {
      switch (format)
      {
      case Format::Undefined:
        return 0;
      case Format::Float:
      case Format::Half:
        return 1;
      case Format::Float2:
      case Format::Half2:
        return 2;
      case Format::Float3:
      case Format::Half3:
        return 3;
      case Format::Float4:
      case Format::Half4:
        return 4;
      default:
        throw Exception(Error::InvalidArgument, "invalid image format");
      }
    }

    // Returns the height of the image
    oidn_inline int getH() const { return int(height); }

    // Returns the width of the image
    oidn_inline int getW() const { return int(width); }

    // Returns the number of pixels in the image
    oidn_inline size_t getNumElements() const { return width * height; }

    // Returns the size in bytes of the image
    oidn_inline size_t getByteSize() const
    {
      if (width == 0 || height == 0)
        return 0;
      return (height - 1) * hByteStride + (width - 1) * wByteStride + getFormatSize(format);
    }

    oidn_inline DataType getDataType() const
    {
      return getFormatDataType(format);
    }
  };

  class Image final : public Memory, private ImageDesc
  {
  public:
    Image();
    Image(void* ptr, Format format, size_t width, size_t height, size_t byteOffset, size_t pixelByteStride, size_t rowByteStride);
    Image(const Ref<Buffer>& buffer, const ImageDesc& desc, size_t byteOffset);
    Image(const Ref<Buffer>& buffer, Format format, size_t width, size_t height, size_t byteOffset, size_t pixelByteStride, size_t rowByteStride);
    Image(Engine* engine, Format format, size_t width, size_t height);

    void postRealloc() override;

    oidn_inline const ImageDesc& getDesc() const { return *this; }
    oidn_inline Format getFormat() const { return format; }

    using ImageDesc::getC;
    using ImageDesc::getH;
    using ImageDesc::getW;
    using ImageDesc::getNumElements;
    using ImageDesc::getByteSize;
    using ImageDesc::getDataType;

    oidn_inline void* getPtr() const { return ptr; }
    oidn_inline operator bool() const { return ptr || buffer; }

    operator ImageAccessor()
    {
      ImageAccessor acc;
      acc.ptr = ptr;
      acc.hByteStride = hByteStride;
      acc.wByteStride = wByteStride;
      acc.dataType = getDataType();
      acc.C = getC();
      if (acc.C > 3)
        throw std::logic_error("unsupported number of channels for image accessor");
      acc.H = getH();
      acc.W = getW();
      return acc;
    }

    operator ispc::ImageAccessor();

    // Determines whether two images overlap in memory
    bool overlaps(const Image& other) const;

  private:
    char* ptr; // pointer to the first pixel
  };

OIDN_NAMESPACE_END
