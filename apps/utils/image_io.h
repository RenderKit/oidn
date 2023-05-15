// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image_buffer.h"

OIDN_NAMESPACE_BEGIN

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

OIDN_NAMESPACE_END