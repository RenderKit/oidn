// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "buffer.h"
#include "image_accessor.h"

namespace oidn {

  struct ImageDesc
  {
    static constexpr size_t maxDim = 65536;

    size_t width;   // width in number of pixels
    size_t height;  // height in number of pixels
    size_t wStride; // pixel stride in number of bytes
    size_t hStride; // row stride in number of bytes
    Format format;  // pixel format

    ImageDesc() = default;
    ImageDesc(Format format, size_t width, size_t height, size_t bytePixelStride = 0, size_t byteRowStride = 0);

    // Returns the number of channels
    OIDN_INLINE int getC() const
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
    OIDN_INLINE int getH() const { return int(height); }

    // Returns the width of the image
    OIDN_INLINE int getW() const { return int(width); }

    // Returns the number of pixels in the image
    OIDN_INLINE size_t getNumElements() const { return width * height; }

    // Returns the size in bytes of the image
    OIDN_INLINE size_t getByteSize() const { return height * hStride; }

    // Returns the aligned size in bytes of the image
    OIDN_INLINE size_t getAlignedSize() const
    {
      return round_up(getByteSize(), memoryAlignment);
    }

    OIDN_INLINE DataType getDataType() const
    {
      return getFormatDataType(format);
    }
  };

  class Image : public Memory
  {
  private:
    ImageDesc desc;
    char* ptr; // pointer to the first pixel

  public:
    Image();
    Image(void* ptr, Format format, size_t width, size_t height, size_t byteOffset, size_t bytePixelStride, size_t byteRowStride);
    Image(const Ref<Buffer>& buffer, const ImageDesc& desc, size_t byteOffset);
    Image(const Ref<Buffer>& buffer, Format format, size_t width, size_t height, size_t byteOffset, size_t bytePixelStride, size_t byteRowStride);
    Image(const Ref<Device>& device, Format format, size_t width, size_t height);

    void updatePtr() override;

    OIDN_INLINE const ImageDesc& getDesc() const { return desc; }
    OIDN_INLINE Format getFormat() const { return desc.format; }
    OIDN_INLINE int getC() const { return desc.getC(); }
    OIDN_INLINE int getH() const { return desc.getH(); }
    OIDN_INLINE int getW() const { return desc.getW(); }
    OIDN_INLINE size_t getNumElements() const { return desc.getNumElements(); }
    OIDN_INLINE size_t getByteSize() const { return desc.getByteSize(); }
    OIDN_INLINE size_t getAlignedSize() const { return desc.getAlignedSize(); }
    OIDN_INLINE DataType getDataType() const { return desc.getDataType(); }

    OIDN_INLINE       char* begin()       { return ptr; }
    OIDN_INLINE const char* begin() const { return ptr; }

    OIDN_INLINE       char* end()       { return ptr + getByteSize(); }
    OIDN_INLINE const char* end() const { return ptr + getByteSize(); }

    OIDN_INLINE operator bool() const { return ptr != nullptr; }

    template<typename T>
    operator ImageAccessor<T>() const
    {
      if (desc.format != Format::Undefined && getDataType() != DataTypeOf<T>::value)
        throw Exception(Error::Unknown, "incompatible image accessor");

      ImageAccessor<T> acc;
      acc.ptr = (uint8_t*)ptr;
      acc.hStride = desc.hStride;
      acc.wStride = desc.wStride;
      acc.W = desc.width;
      acc.H = desc.height;
      return acc;
    }

    operator ispc::ImageAccessor() const;

    // Determines whether two images overlap in memory
    bool overlaps(const Image& other) const;
  };

} // namespace oidn
