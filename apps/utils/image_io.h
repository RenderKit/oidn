// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <OpenImageDenoise/oidn.hpp>
#include "common/platform.h"

namespace oidn {

  class ImageBuffer
  {
  private:
    DeviceRef device;
    BufferRef buffer;
    char* bufferPtr;
    size_t numValues;
    int width;
    int height;
    int numChannels;
    Format dataType;

  public:
    ImageBuffer()
      : bufferPtr(nullptr),
        numValues(0),
        width(0),
        height(0),
        numChannels(0),
        dataType(Format::Undefined) {}

    ImageBuffer(const DeviceRef& device, int width, int height, int numChannels, Format dataType = Format::Float);

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
    
    OIDN_INLINE const void* getData() const { return bufferPtr; }
    OIDN_INLINE void* getData() { return bufferPtr; }

    OIDN_INLINE operator bool() const { return getData() != nullptr; }

    template<typename T = float>
    T get(size_t i) const;

    OIDN_INLINE void set(size_t i, float x)
    {
      switch (dataType)
      {
      case Format::Float:
        ((float*)bufferPtr)[i] = x;
        break;
      case Format::Half:
        ((half*)bufferPtr)[i] = half(x);
        break;
      default:
        assert(0);
      }
    }

    OIDN_INLINE void set(size_t i, half x)
    {
      switch (dataType)
      {
      case Format::Float:
        ((float*)bufferPtr)[i] = float(x);
        break;
      case Format::Half:
        ((half*)bufferPtr)[i] = x;
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
  };

  template<>
  OIDN_INLINE float ImageBuffer::get(size_t i) const
  {
    switch (dataType)
    {
    case Format::Float:
      return ((float*)bufferPtr)[i];
    case Format::Half:
      return float(((half*)bufferPtr)[i]);
    default:
      assert(0);
      return 0;
    }
  }

  template<>
  OIDN_INLINE half ImageBuffer::get(size_t i) const
  {
    switch (dataType)
    {
    case Format::Float:
      return half(((float*)bufferPtr)[i]);
    case Format::Half:
      return ((half*)bufferPtr)[i];
    default:
      assert(0);
      return 0;
    }
  }

  // Loads an image with optionally specified number of channels and data type
  std::shared_ptr<ImageBuffer> loadImage(const DeviceRef& device,
                                         const std::string& filename,
                                         int numChannels = 0,
                                         Format dataType = Format::Undefined);

  // Loads an image with/without sRGB to linear conversion
  std::shared_ptr<ImageBuffer> loadImage(const DeviceRef& device,
                                         const std::string& filename,
                                         int numChannels,
                                         bool srgb,
                                         Format dataType = Format::Undefined);

  // Saves an image
  void saveImage(const std::string& filename, const ImageBuffer& image);

  // Saves an image with/without linear to sRGB conversion
  void saveImage(const std::string& filename, const ImageBuffer& image, bool srgb);

  // Compares an image to a reference image and returns the number of errors
  // and the maximum error value
  std::tuple<size_t, float> compareImage(const ImageBuffer& image,
                                         const ImageBuffer& ref,
                                         float threshold);

} // namespace oidn
