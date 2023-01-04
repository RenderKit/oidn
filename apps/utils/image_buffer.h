// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <tuple>

namespace oidn {

  class ImageBuffer
  {
  public:
    ImageBuffer();
    ImageBuffer(const DeviceRef& device, int width, int height, int numChannels, Format dataType = Format::Float, Storage storage = Storage::Undefined);

    OIDN_INLINE int getW() const { return width; }
    OIDN_INLINE int getH() const { return height; }
    OIDN_INLINE int getC() const { return numChannels; }
    std::array<int, 3> getDims() const { return {width, height, numChannels}; }

    OIDN_INLINE Format getDataType() const { return dataType; }

    Format getFormat() const
    {
      if (dataType == Format::Undefined)
        return Format::Undefined;
      return Format(int(dataType) + numChannels - 1);
    }

    OIDN_INLINE size_t getSize() const { return numValues; }
    OIDN_INLINE size_t getByteSize() const { return buffer.getSize(); }
    
    OIDN_INLINE const void* getData() const { return hostPtr ? hostPtr : devPtr; }
    OIDN_INLINE void* getData() { return hostPtr ? hostPtr : devPtr; }

    OIDN_INLINE const BufferRef& getBuffer() const { return buffer; }

    OIDN_INLINE operator bool() const { return devPtr != nullptr; }

    void map(Access access);
    void unmap();

    template<typename T = float>
    T get(size_t i) const;

    OIDN_INLINE void set(size_t i, float x)
    {
      assert(hostPtr);

      switch (dataType)
      {
      case Format::Float:
        ((float*)hostPtr)[i] = x;
        break;
      case Format::Half:
        ((half*)hostPtr)[i] = half(x);
        break;
      default:
        assert(0);
      }
    }

    OIDN_INLINE void set(size_t i, half x)
    {
      assert(hostPtr);

      switch (dataType)
      {
      case Format::Float:
        ((float*)hostPtr)[i] = float(x);
        break;
      case Format::Half:
        ((half*)hostPtr)[i] = x;
        break;
      default:
        assert(0);
      }
    }

    // Returns a copy of the image buffer
    std::shared_ptr<ImageBuffer> clone() const;

  private:
    // Disable copying
    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator =(const ImageBuffer&) = delete;

    DeviceRef device;
    BufferRef buffer;
    char* devPtr;
    char* hostPtr;
    size_t numValues;
    int width;
    int height;
    int numChannels;
    Format dataType;
  };

  template<>
  OIDN_INLINE float ImageBuffer::get(size_t i) const
  {
    assert(hostPtr);

    switch (dataType)
    {
    case Format::Float:
      return ((float*)hostPtr)[i];
    case Format::Half:
      return float(((half*)hostPtr)[i]);
    default:
      assert(0);
      return 0;
    }
  }

  template<>
  OIDN_INLINE half ImageBuffer::get(size_t i) const
  {
    assert(hostPtr);

    switch (dataType)
    {
    case Format::Float:
      return half(((float*)hostPtr)[i]);
    case Format::Half:
      return ((half*)hostPtr)[i];
    default:
      assert(0);
      return 0;
    }
  }

  // Compares an image to a reference image and returns the number of errors
  // and the maximum error value
  std::tuple<size_t, float> compareImage(const ImageBuffer& image,
                                         const ImageBuffer& ref,
                                         float threshold);

} // namespace oidn
