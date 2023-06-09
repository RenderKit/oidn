// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include <memory>
#include <string>
#include <vector>
#include <array>
#include <tuple>

OIDN_NAMESPACE_BEGIN

  class ImageBuffer
  {
  public:
    ImageBuffer();
    ImageBuffer(const DeviceRef& device, int width, int height, int numChannels,
                DataType dataType = DataType::Float32,
                Storage storage = Storage::Undefined,
                bool forceHostCopy = false);
    ~ImageBuffer();

    OIDN_INLINE int getW() const { return width; }
    OIDN_INLINE int getH() const { return height; }
    OIDN_INLINE int getC() const { return numChannels; }
    std::array<int, 3> getDims() const { return {width, height, numChannels}; }

    OIDN_INLINE DataType getDataType() const { return dataType; }
    OIDN_INLINE Format getFormat() const { return format; }

    OIDN_INLINE size_t getSize() const { return numValues; }
    OIDN_INLINE size_t getByteSize() const { return byteSize; }

    OIDN_INLINE const void* getData() const { return devPtr; }
    OIDN_INLINE void* getData() { return devPtr; }

    OIDN_INLINE const BufferRef& getBuffer() const { return buffer; }

    OIDN_INLINE operator bool() const { return devPtr != nullptr; }

    // Data with device storage must be explicitly copied between host and device
    void toHost();
    void toHostAsync();
    void toDevice();
    void toDeviceAsync();

    template<typename T = float>
    T get(size_t i) const;

    OIDN_INLINE void set(size_t i, float x)
    {
      switch (dataType)
      {
      case DataType::Float32:
        reinterpret_cast<float*>(hostPtr)[i] = x;
        break;
      case DataType::Float16:
        reinterpret_cast<half*>(hostPtr)[i] = half(x);
        break;
      default:
        assert(0);
      }
    }

    OIDN_INLINE void set(size_t i, half x)
    {
      switch (dataType)
      {
      case DataType::Float32:
        reinterpret_cast<float*>(hostPtr)[i] = float(x);
        break;
      case DataType::Float16:
        reinterpret_cast<half*>(hostPtr)[i] = x;
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
    size_t byteSize;
    size_t numValues;
    int width;
    int height;
    int numChannels;
    DataType dataType;
    Format format;
  };

  template<>
  OIDN_INLINE float ImageBuffer::get(size_t i) const
  {
    switch (dataType)
    {
    case DataType::Float32:
      return reinterpret_cast<float*>(hostPtr)[i];
    case DataType::Float16:
      return float(reinterpret_cast<half*>(hostPtr)[i]);
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
    case DataType::Float32:
      return half(reinterpret_cast<float*>(hostPtr)[i]);
    case DataType::Float16:
      return reinterpret_cast<half*>(hostPtr)[i];
    default:
      assert(0);
      return 0;
    }
  }

  // Compares an image to a reference image and returns the number of errors
  // and the average error value
  std::tuple<size_t, double> compareImage(const ImageBuffer& image,
                                          const ImageBuffer& ref,
                                          double errorThreshold = 0.005);

OIDN_NAMESPACE_END
