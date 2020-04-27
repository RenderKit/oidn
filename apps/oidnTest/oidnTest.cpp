// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <limits>
#include <OpenImageDenoise/oidn.hpp>
#include "apps/utils/image_io.h"

#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_FAST_COMPILE
#include "catch.hpp"

using namespace oidn;

void setFilterImage(FilterRef& filter, const char* name, ImageBuffer& image)
{
  Format format = Format::Undefined;
  switch (image.getChannels())
  {
  case 1: format = Format::Float;  break;
  case 2: format = Format::Float2; break;
  case 3: format = Format::Float3; break;
  case 4: format = Format::Float4; break;
  default: assert(0);
  }

  filter.setImage(name, image.getData(), format, image.getWidth(), image.getHeight());
}

ImageBuffer makeConstImage(int W, int H, int C, float value)
{
  ImageBuffer image(W, H, C);
  for (int i = 0; i < image.getSize(); ++i)
    image[i] = value;
  return image;
}

bool isBetween(const ImageBuffer& image, float a, float b)
{
  for (int i = 0; i < image.getSize(); ++i)
    if (!std::isfinite(image[i]) || image[i] < a || image[i] > b)
      return false;
  return true;
}

void sanitizationTest(DeviceRef& device, bool hdr, float value)
{
  const int W = 190;
  const int H = 347;

  FilterRef filter = device.newFilter("RT");
  REQUIRE(filter);

  ImageBuffer input = makeConstImage(W, H, 3, value);
  ImageBuffer output(W, H, 3);
  setFilterImage(filter, "color",  input);
  setFilterImage(filter, "albedo", input);
  setFilterImage(filter, "normal", input);
  setFilterImage(filter, "output", output);
  filter.set("hdr", hdr);
  filter.commit();

  filter.execute();
  REQUIRE(device.getError() == Error::None);

  if (hdr)
    REQUIRE(isBetween(output, 0.f, std::numeric_limits<float>::max()));
  else
    REQUIRE(isBetween(output, 0.f, 1.f));
}

TEST_CASE("image sanitization", "[sanitization]")
{
  DeviceRef device = oidn::newDevice();
  device.commit();
  REQUIRE(device);

  SECTION("HDR")
  {
    sanitizationTest(device, true,  std::numeric_limits<float>::quiet_NaN());
    sanitizationTest(device, true,  std::numeric_limits<float>::infinity());
    sanitizationTest(device, true, -std::numeric_limits<float>::infinity());
    sanitizationTest(device, true, -100.f);
  }

  SECTION("LDR")
  {
    sanitizationTest(device, false,  std::numeric_limits<float>::quiet_NaN());
    sanitizationTest(device, false,  std::numeric_limits<float>::infinity());
    sanitizationTest(device, false,  10.f);
    sanitizationTest(device, false, -2.f);
  }
}