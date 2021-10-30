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
  public:
    DeviceRef device;
    BufferRef buffer;
    char* bufferPtr;
    size_t numValues;
    int width;
    int height;
    int numChannels;
    Format dataType;

    ImageBuffer()
      : bufferPtr(nullptr),
        numValues(0),
        width(0),
        height(0),
        numChannels(0),
        dataType(Format::Undefined) {}

    ImageBuffer(const DeviceRef& device, int width, int height, int numChannels, Format dataType = Format::Float)
      : device(device),
        numValues(size_t(width) * height * numChannels),
        width(width),
        height(height),
        numChannels(numChannels),
        dataType(dataType)
    {
      size_t valueSize = 0;
      switch (dataType)
      {
      case Format::Float: valueSize = sizeof(float);   break;
      case Format::Half:  valueSize = sizeof(int16_t); break;
      default:            assert(0);
      }
    
      buffer = device.newBuffer(numValues * valueSize);
      bufferPtr = (char*)buffer.getData();
    }

    operator bool() const
    {
      return data() != nullptr;
    }

    template<typename T = float>
    T get(size_t i) const;

    // Float
    __forceinline void set(size_t i, float x)
    {
      switch (dataType)
      {
      case Format::Float:
        ((float*)bufferPtr)[i] = x;
        break;
      case Format::Half:
        ((int16_t*)bufferPtr)[i] = float_to_half(x);
        break;
      default:
        assert(0);
      }
    }

    // Half
    __forceinline void set(size_t i, int16_t x)
    {
      switch (dataType)
      {
      case Format::Float:
        ((float*)bufferPtr)[i] = half_to_float(x);
        break;
      case Format::Half:
        ((int16_t*)bufferPtr)[i] = x;
        break;
      default:
        assert(0);
      }
    }

    __forceinline const void* data() const { return bufferPtr; }
    __forceinline void* data() { return bufferPtr; }
  
    __forceinline size_t size() const { return numValues; }
    std::array<int, 3> dims() const { return {width, height, numChannels}; }
    size_t byteSize() const { return buffer.getSize(); }

    Format format() const
    {
      if (dataType == Format::Undefined)
        return Format::Undefined;
      return Format(int(dataType) + numChannels - 1);
    }

    // Returns a copy of the image buffer
    std::shared_ptr<ImageBuffer> clone() const
    {
      auto result = std::make_shared<ImageBuffer>(device, width, height, numChannels, dataType);
      memcpy(result->bufferPtr, bufferPtr, byteSize());
      return result;
    }

  private:
    // Disable copying
    ImageBuffer(const ImageBuffer&) = delete;
    ImageBuffer& operator =(const ImageBuffer&) = delete;
  };

  // Float
  template<>
  __forceinline float ImageBuffer::get(size_t i) const
  {
    switch (dataType)
    {
    case Format::Float:
      return ((float*)bufferPtr)[i];
    case Format::Half:
      return half_to_float(((int16_t*)bufferPtr)[i]);
    default:
      assert(0);
      return 0;
    }
  }

  // Half
  template<>
  __forceinline int16_t ImageBuffer::get(size_t i) const
  {
    switch (dataType)
    {
    case Format::Float:
      return float_to_half(((float*)bufferPtr)[i]);
    case Format::Half:
      return ((int16_t*)bufferPtr)[i];
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
