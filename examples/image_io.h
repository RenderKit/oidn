// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>
#include <array>

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
    std::array<int, 2> getSize() const { return {width, height}; }
    int getChannels() const { return channels; }

    const float* getData() const { return data.data(); }
    float* getData() { return data.data(); }
    int getDataSize() { return int(data.size()); }
  };

  ImageBuffer loadImage(const std::string& filename);
  void saveImage(const std::string& filename, const ImageBuffer& image);

} // namespace oidn
