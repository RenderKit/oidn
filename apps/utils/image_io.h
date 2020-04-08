// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <array>
#include <tuple>

namespace oidn {

  class ImageBuffer
  {
  private:
    std::vector<float> data;
    int width;
    int height;
    int channels;

  public:
    ImageBuffer()
      : width(0),
        height(0),
        channels(0) {}

    ImageBuffer(int width, int height, int channels)
      : data(width * height * channels),
      width(width),
        height(height),
        channels(channels) {}

    operator bool() const
    {
      return data.data() != nullptr;
    }

    const float& operator [](size_t i) const { return data[i]; }
    float& operator [](size_t i) { return data[i]; }

    int getWidth() const { return width; }
    int getHeight() const { return height; }
    std::array<int, 3> getDims() const { return {width, height, channels}; }
    int getChannels() const { return channels; }

    const float* getData() const { return data.data(); }
    float* getData() { return data.data(); }
    int getSize() const { return int(data.size()); }
  };

  // Loads an image with an optionally specified number of channels (loads all
  // channels by default)
  ImageBuffer loadImage(const std::string& filename, int channels = 0);

  // Loads an image with/without sRGB to linear conversion
  ImageBuffer loadImage(const std::string& filename, int channels, bool srgb);

  // Saves an image
  void saveImage(const std::string& filename, const ImageBuffer& image);

  // Saves an image with/without linear to sRGB conversion
  void saveImage(const std::string& filename, const ImageBuffer& image, bool srgb);

  // Compares an image to a reference image and returns the number of errors
  // and the maximum error value
  std::tuple<int, float> compareImage(const ImageBuffer& image,
                                      const ImageBuffer& ref,
                                      float threshold);

} // namespace oidn
