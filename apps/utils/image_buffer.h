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

    oidn_inline int getW() const { return width; }
    oidn_inline int getH() const { return height; }
    oidn_inline int getC() const { return numChannels; }
    std::array<int, 3> getDims() const { return {width, height, numChannels}; }

    oidn_inline DataType getDataType() const { return dataType; }
    oidn_inline Format getFormat() const { return format; }

    oidn_inline size_t getSize() const { return numValues; }
    oidn_inline size_t getByteSize() const { return byteSize; }

    oidn_inline const void* getData() const { return devPtr; }
    oidn_inline void* getData() { return devPtr; }
    oidn_inline const void* getHostData() const { return hostPtr; }
    oidn_inline void* getHostData() { return hostPtr; }

    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr) const;
    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr);

    oidn_inline const BufferRef& getBuffer() const { return buffer; }

    oidn_inline operator bool() const { return buffer; }

    // Data with device storage must be explicitly copied between host and device
    void toHost();
    void toHostAsync();
    void toDevice();
    void toDeviceAsync();

    template<typename T = float>
    T get(size_t i) const;

    oidn_inline void set(size_t i, float x)
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

    oidn_inline void set(size_t i, half x)
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
  oidn_inline float ImageBuffer::get(size_t i) const
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
  oidn_inline half ImageBuffer::get(size_t i) const
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

  // Compares an image to a reference image and returns whether they match exactly
  bool compareImage(const ImageBuffer& image, const ImageBuffer& ref);

  // Compares an image to a reference image and returns the number of errors
  // and the average error value
  std::tuple<size_t, double> compareImage(const ImageBuffer& image,
                                          const ImageBuffer& ref,
                                          double errorThreshold);

OIDN_NAMESPACE_END
