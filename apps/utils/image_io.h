// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <array>
#include <tuple>
#include <OpenImageDenoise/oidn.hpp>

namespace oidn {

  struct ImageBuffer
  {
    BufferRef buffer;
    float* bufferPtr;
    size_t bufferSize;
    int width;
    int height;
    int numChannels;

    ImageBuffer()
      : bufferPtr(nullptr),
        bufferSize(0),
        width(0),
        height(0),
        numChannels(0) {}

    ImageBuffer(DeviceRef& device, int width, int height, int numChannels)
      : bufferSize(size_t(width) * height * numChannels),
        width(width),
        height(height),
        numChannels(numChannels)
    {
      buffer = device.newBuffer(bufferSize * sizeof(float));
      bufferPtr = (float*)buffer.getData();
    }

    operator bool() const
    {
      return data() != nullptr;
    }

    const float& operator [](size_t i) const { return bufferPtr[i]; }
    float& operator [](size_t i) { return bufferPtr[i]; }

    const float* data() const { return bufferPtr; }
    float* data() { return bufferPtr; }
  
    size_t size() const { return bufferSize; }
    std::array<int, 3> dims() const { return {width, height, numChannels}; }
  };

  // Loads an image with an optionally specified number of channels (loads all
  // channels by default)
  std::shared_ptr<ImageBuffer> loadImage(DeviceRef& device, const std::string& filename, int numChannels = 0);

  // Loads an image with/without sRGB to linear conversion
  std::shared_ptr<ImageBuffer> loadImage(DeviceRef& device, const std::string& filename, int numChannels, bool srgb);

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
