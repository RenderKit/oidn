// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <cmath>
#include <limits>
#include <OpenImageDenoise/oidn.hpp>
#include "apps/utils/image_io.h"

#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_FAST_COMPILE
#include "catch.hpp"

OIDN_NAMESPACE_USING
using namespace oidn;

#if defined(OIDN_FILTER_RT)

void setFilterImage(FilterRef& filter, const char* name, ImageBuffer& image)
{
  Format format = Format::Undefined;
  switch (image.numChannels)
  {
  case 1: format = Format::Float;  break;
  case 2: format = Format::Float2; break;
  case 3: format = Format::Float3; break;
  case 4: format = Format::Float4; break;
  default:
    assert(0);
  }

  filter.setImage(name, image.data(), format, image.width, image.height);
}

ImageBuffer makeConstImage(int W, int H, int C = 3, float value = 0.5f)
{
  ImageBuffer image(W, H, C);
  for (size_t i = 0; i < image.size(); ++i)
    image[i] = value;
  return image;
}

bool isBetween(const ImageBuffer& image, float a, float b)
{
  for (size_t i = 0; i < image.size(); ++i)
    if (!std::isfinite(image[i]) || image[i] < a || image[i] > b)
      return false;
  return true;
}

// -----------------------------------------------------------------------------

TEST_CASE("single filter", "[single_filter]")
{
  const int W = 257;
  const int H = 89;

  DeviceRef device = newDevice();
  REQUIRE(device);
  device.commit();
  REQUIRE(device.getError() == Error::None);

  FilterRef filter = device.newFilter("RT");
  REQUIRE(filter);

  ImageBuffer image = makeConstImage(W, H);
  setFilterImage(filter, "color",  image);
  setFilterImage(filter, "output", image);

  filter.commit();
  REQUIRE(device.getError() == Error::None);

  SECTION("single filter: 1 frame")
  {
    filter.execute();
    REQUIRE(device.getError() == Error::None);
  }

  SECTION("single filter: 3 frames")
  {
    for (int i = 0; i < 3; ++i)
    {
      filter.execute();
      REQUIRE(device.getError() == Error::None);
    }
  }
}

// -----------------------------------------------------------------------------

void multiFilter1PerDeviceTest(DeviceRef& device, const std::vector<int>& sizes)
{
  for (size_t i = 0; i < sizes.size(); ++i)
  {
    FilterRef filter = device.newFilter("RT");
    REQUIRE(filter);

    ImageBuffer image = makeConstImage(sizes[i], sizes[i]);
    setFilterImage(filter, "color",  image);
    setFilterImage(filter, "output", image);

    filter.commit();
    REQUIRE(device.getError() == Error::None);

    filter.execute();
    REQUIRE(device.getError() == Error::None);
  }
}

void multiFilterNPerDeviceTest(DeviceRef& device, const std::vector<int>& sizes)
{
  std::vector<FilterRef> filters;
  std::vector<ImageBuffer> images;

  for (size_t i = 0; i < sizes.size(); ++i)
  {
    filters.push_back(device.newFilter("RT"));
    REQUIRE(filters[i]);

    images.push_back(makeConstImage(sizes[i], sizes[i]));
    setFilterImage(filters[i], "color",  images[i]);
    setFilterImage(filters[i], "output", images[i]);

    filters[i].commit();
    REQUIRE(device.getError() == Error::None);
  }

  for (size_t i = 0; i < filters.size(); ++i)
  {
    filters[i].execute();
    REQUIRE(device.getError() == Error::None);
  }
}

TEST_CASE("multiple filters", "[multi_filter]")
{
  DeviceRef device = newDevice();
  REQUIRE(device);
  device.commit();
  REQUIRE(device.getError() == Error::None);

  SECTION("1 filter / device: small -> large -> medium")
  {
    multiFilter1PerDeviceTest(device, {257, 3001, 1024});
  }

  SECTION("3 filters / device: small")
  {
    multiFilterNPerDeviceTest(device, {256, 256, 256});
  }

  SECTION("2 filters / device: small -> large")
  {
    multiFilterNPerDeviceTest(device, {256, 3001});
  }

  SECTION("3 filters / device: large -> small -> medium")
  {
    multiFilterNPerDeviceTest(device, {3001, 257, 1024});
  }
}

// -----------------------------------------------------------------------------

TEST_CASE("multiple devices", "[multi_device]")
{
  const std::vector<int> sizes = {111, 256, 80};

  std::vector<DeviceRef> devices;
  std::vector<FilterRef> filters;
  std::vector<ImageBuffer> images;

  for (size_t i = 0; i < sizes.size(); ++i)
  {
    devices.push_back(newDevice());
    REQUIRE(devices[i]);
    devices[i].commit();
    REQUIRE(devices[i].getError() == Error::None);

    filters.push_back(devices[i].newFilter("RT"));
    REQUIRE(filters[i]);

    images.push_back(makeConstImage(sizes[i], sizes[i]));
    setFilterImage(filters[i], "color",  images[i]);
    setFilterImage(filters[i], "output", images[i]);

    filters[i].commit();
    REQUIRE(devices[i].getError() == Error::None);
  }

  for (size_t i = 0; i < devices.size(); ++i)
  {
    filters[i].execute();
    REQUIRE(devices[i].getError() == Error::None);
  }
}

// -----------------------------------------------------------------------------

TEST_CASE("filter update", "[filter_update]")
{
  const int W = 211;
  const int H = 599;

  DeviceRef device = newDevice();
  REQUIRE(device);
  device.commit();
  REQUIRE(device.getError() == Error::None);

  FilterRef filter = device.newFilter("RT");
  REQUIRE(filter);

  ImageBuffer color  = makeConstImage(W, H);
  ImageBuffer albedo = makeConstImage(W, H);
  ImageBuffer output = makeConstImage(W, H);
  setFilterImage(filter, "color",  color);
  setFilterImage(filter, "albedo", albedo);
  setFilterImage(filter, "output", output);

  filter.set("hdr", true);

  filter.commit();
  REQUIRE(device.getError() == Error::None);

  filter.execute();
  REQUIRE(device.getError() == Error::None);

  SECTION("filter update: none")
  {
    // No changes
  }

  SECTION("filter update: same image size")
  {
    color = makeConstImage(W, H);
    setFilterImage(filter, "color", color);
  }

  SECTION("filter update: different image size")
  {
    color  = makeConstImage(W*2, H*2);
    albedo = makeConstImage(W*2, H*2);
    output = makeConstImage(W*2, H*2);
    setFilterImage(filter, "color",  color);
    setFilterImage(filter, "albedo", albedo);
    setFilterImage(filter, "output", output);
  }

  SECTION("filter update: remove image")
  {
    filter.removeImage("albedo");
  }

  SECTION("filter update: remove image by setting to null")
  {
    filter.setImage("albedo", nullptr, Format::Float3, 0, 0);
  }

  SECTION("filter update: different mode")
  {
    filter.set("hdr", false);
  }

  filter.commit();
  REQUIRE(device.getError() == Error::None);

  filter.execute();
  REQUIRE(device.getError() == Error::None);
}

// -----------------------------------------------------------------------------

void imageSizeTest(DeviceRef& device, int W, int H)
{
  FilterRef filter = device.newFilter("RT");
  REQUIRE(filter);

  const int N = std::max(W * H * 3, 1); // make sure the buffers are never null
  std::vector<float> input(N, 0.5f);
  std::vector<float> output(N);

  filter.setImage("color",  input.data(),  Format::Float3, W, H);
  filter.setImage("output", output.data(), Format::Float3, W, H);

  filter.commit();
  REQUIRE(device.getError() == Error::None);

  filter.execute();
  REQUIRE(device.getError() == Error::None);
}

TEST_CASE("image size", "[size]")
{
  DeviceRef device = newDevice();
  REQUIRE(device);
  device.commit();
  REQUIRE(device.getError() == Error::None);

  SECTION("image size: 0x0")
  {
    imageSizeTest(device, 0, 0);
  }

  SECTION("image size: [0,1]x[0,1]")
  {
    imageSizeTest(device, 0, 1);
    imageSizeTest(device, 1, 0);
  }

  SECTION("image size: [1,2]x[1,2]")
  {
    imageSizeTest(device, 1, 1);
    imageSizeTest(device, 1, 2);
    imageSizeTest(device, 2, 1);
    imageSizeTest(device, 2, 2);
  }
}

// -----------------------------------------------------------------------------

void sanitizationTest(DeviceRef& device, bool hdr, float value)
{
  const int W = 191;
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
  REQUIRE(device.getError() == Error::None);

  filter.execute();
  REQUIRE(device.getError() == Error::None);

  if (hdr)
    REQUIRE(isBetween(output, 0.f, std::numeric_limits<float>::max()));
  else
    REQUIRE(isBetween(output, 0.f, 1.f));
}

TEST_CASE("image sanitization", "[sanitization]")
{
  DeviceRef device = newDevice();
  REQUIRE(device);
  device.commit();
  REQUIRE(device.getError() == Error::None);

  SECTION("image sanitization: HDR")
  {
    sanitizationTest(device, true,  std::numeric_limits<float>::quiet_NaN());
    sanitizationTest(device, true,  std::numeric_limits<float>::infinity());
    sanitizationTest(device, true, -std::numeric_limits<float>::infinity());
    sanitizationTest(device, true, -100.f);
  }

  SECTION("image sanitization: LDR")
  {
    sanitizationTest(device, false,  std::numeric_limits<float>::quiet_NaN());
    sanitizationTest(device, false,  std::numeric_limits<float>::infinity());
    sanitizationTest(device, false,  10.f);
    sanitizationTest(device, false, -2.f);
  }
}

// -----------------------------------------------------------------------------

struct Progress
{
  double n;    // current progress
  double nMax; // when to cancel execution

  Progress(double nMax) : n(0), nMax(nMax) {}
};

// Progress monitor callback function
bool progressCallback(void* userPtr, double n)
{
  Progress* progress = (Progress*)userPtr;
  REQUIRE((std::isfinite(n) && n >= 0 && n <= 1)); // n must be between 0 and 1
  REQUIRE(n >= progress->n);   // n must not decrease
  progress->n = n;
  return n < progress->nMax; // cancel if reached nMax
}

void progressTest(DeviceRef& device, double nMax = 1000)
{
  const int W = 1283;
  const int H = 727;

  FilterRef filter = device.newFilter("RT");
  REQUIRE(filter);

  ImageBuffer image = makeConstImage(W, H);
  setFilterImage(filter, "color",  image);
  setFilterImage(filter, "output", image); // in-place

  Progress progress(nMax);
  filter.setProgressMonitorFunction(progressCallback, &progress);

  filter.set("maxMemoryMB", 0); // make sure there will be multiple tiles

  filter.commit();
  REQUIRE(device.getError() == Error::None);

  filter.execute();

  if (nMax <= 1)
  {
    // Execution should be cancelled
    REQUIRE(device.getError() == Error::Cancelled);
    REQUIRE(progress.n >= nMax); // check whether execution was cancelled too early
  }
  else
  {
    // Execution should be finished
    REQUIRE(device.getError() == Error::None);
    REQUIRE(progress.n == 1); // progress must be 100% at the end
  }
}

TEST_CASE("progress monitor", "[progress]")
{
  DeviceRef device = newDevice();
  REQUIRE(device);
  device.commit();
  REQUIRE(device.getError() == Error::None);

  SECTION("progress monitor: complete")
  {
    progressTest(device);
  }

  SECTION("progress monitor: cancel at the middle")
  {
    progressTest(device, 0.5);
  }

  SECTION("progress monitor: cancel at the beginning")
  {
    progressTest(device, 0);
  }
 
  SECTION("progress monitor: cancel at the end")
  {
    progressTest(device, 1);
  }
}

#endif // defined(OIDN_FILTER_RT)