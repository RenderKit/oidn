// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common/common.h"
#include "utils/image_buffer.h"
#include <cassert>
#include <cmath>
#include <limits>

#define CATCH_CONFIG_RUNNER
#define CATCH_CONFIG_FAST_COMPILE
#include "catch.hpp"

OIDN_NAMESPACE_USING

DeviceType deviceType = DeviceType::Default;
PhysicalDeviceRef physicalDevice;

DeviceRef makeDevice()
{
  return physicalDevice ? physicalDevice.newDevice() : newDevice(deviceType);
}

// -------------------------------------------------------------------------------------------------

TEST_CASE("physical device", "[physical_device]")
{
  const int numDevices = getNumPhysicalDevices();
  REQUIRE(getError() == Error::None);
  REQUIRE(numDevices >= 0);

  for (int i = 0; i < numDevices; ++i)
  {
    PhysicalDeviceRef physicalDevice(i);

    const DeviceType type = physicalDevice.get<DeviceType>("type");
    REQUIRE(getError() == Error::None);
    REQUIRE(type != DeviceType::Default);

    physicalDevice.get<int>("qwertyuiop");
    REQUIRE(getError() == Error::InvalidArgument);

    const std::string name = physicalDevice.get<std::string>("name");
    REQUIRE(getError() == Error::None);
    REQUIRE(!name.empty());

    const bool uuidSupported = physicalDevice.get<bool>("uuidSupported");
    REQUIRE(getError() == Error::None);
    physicalDevice.get<OIDN_NAMESPACE::UUID>("uuid");
    if (uuidSupported)
      REQUIRE(getError() == Error::None);
    else
      REQUIRE(getError() == Error::InvalidArgument);

    const bool luidSupported = physicalDevice.get<bool>("luidSupported");
    REQUIRE(getError() == Error::None);

    physicalDevice.get<OIDN_NAMESPACE::LUID>("luid");
    if (luidSupported)
      REQUIRE(getError() == Error::None);
    else
      REQUIRE(getError() == Error::InvalidArgument);

    const uint32_t nodeMask = physicalDevice.get<uint32_t>("nodeMask");
    if (luidSupported)
    {
      REQUIRE(getError() == Error::None);
      REQUIRE(nodeMask != 0);
    }
    else
      REQUIRE(getError() == Error::InvalidArgument);

    const bool pciAddressSupported = physicalDevice.get<bool>("pciAddressSupported");
    REQUIRE(getError() == Error::None);
    physicalDevice.get<int>("pciDomain");
    physicalDevice.get<int>("pciBus");
    physicalDevice.get<int>("pciDevice");
    physicalDevice.get<int>("pciFunction");
    if (pciAddressSupported)
      REQUIRE(getError() == Error::None);
    else
      REQUIRE(getError() == Error::InvalidArgument);

    DeviceRef device = physicalDevice.newDevice();
    REQUIRE(device.getError() == Error::None);
    device.commit();
    REQUIRE(device.getError() == Error::None);
  }
}

// -------------------------------------------------------------------------------------------------

TEST_CASE("buffer", "[buffer]")
{
  DeviceRef device = makeDevice();
  REQUIRE(bool(device));
  device.commit();
  REQUIRE(device.getError() == Error::None);

  SECTION("default buffer")
  {
    BufferRef buffer = device.newBuffer(1234567);
    REQUIRE(device.getError() == Error::None);
  }

  SECTION("device buffer")
  {
    BufferRef buffer = device.newBuffer(1234567, Storage::Device);
    REQUIRE(device.getError() == Error::None);
  }

  SECTION("zero-sized default buffer")
  {
    BufferRef buffer = device.newBuffer(0);
    REQUIRE(device.getError() == Error::None);
  }

  SECTION("zero-sized device buffer")
  {
    BufferRef buffer = device.newBuffer(0, Storage::Device);
    REQUIRE(device.getError() == Error::None);
  }

  SECTION("out-of-memory default buffer")
  {
    BufferRef buffer = device.newBuffer(INTPTR_MAX);
    REQUIRE(device.getError() == Error::OutOfMemory);
  }

  SECTION("out-of-memory device buffer")
  {
    BufferRef buffer = device.newBuffer(INTPTR_MAX, Storage::Device);
    REQUIRE(device.getError() == Error::OutOfMemory);
  }
}

// -------------------------------------------------------------------------------------------------

#if defined(OIDN_FILTER_RT)

void setFilterImage(FilterRef& filter, const char* name, const std::shared_ptr<ImageBuffer>& image,
                    bool useBuffer = false)
{
  Format format = Format::Undefined;
  switch (image->getC())
  {
  case 1: format = Format::Float;  break;
  case 2: format = Format::Float2; break;
  case 3: format = Format::Float3; break;
  case 4: format = Format::Float4; break;
  default:
    assert(0);
  }

  if (useBuffer)
    filter.setImage(name, image->getBuffer(), format, image->getW(), image->getH());
  else
    filter.setImage(name, image->getData(), format, image->getW(), image->getH());
}

std::shared_ptr<ImageBuffer> makeImage(DeviceRef& device, int W, int H, int C = 3)
{
  return std::make_shared<ImageBuffer>(device, W, H, C);
}

std::shared_ptr<ImageBuffer> makeConstImage(DeviceRef& device, int W, int H, int C = 3, float value = 0.5f)
{
  auto image = std::make_shared<ImageBuffer>(device, W, H, C);
  for (size_t i = 0; i < image->getSize(); ++i)
    image->set(i, value);
  return image;
}

bool isBetween(const std::shared_ptr<ImageBuffer>& image, float a, float b)
{
  for (size_t i = 0; i < image->getSize(); ++i)
  {
    const float x = image->get(i);
    if (!std::isfinite(x) || x < a || x > b)
      return false;
  }
  return true;
}

// -------------------------------------------------------------------------------------------------

TEST_CASE("single filter", "[single_filter]")
{
  const int W = 257;
  const int H = 89;

  DeviceRef device = makeDevice();
  REQUIRE(bool(device));
  device.commit();
  REQUIRE(device.getError() == Error::None);

  FilterRef filter = device.newFilter("RT");
  REQUIRE(bool(filter));

  auto image = makeConstImage(device, W, H);
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

// -------------------------------------------------------------------------------------------------

void multiFilter1PerDeviceTest(DeviceRef& device, const std::vector<int>& sizes)
{
  for (size_t i = 0; i < sizes.size(); ++i)
  {
    FilterRef filter = device.newFilter("RT");
    REQUIRE(bool(filter));

    auto image = makeConstImage(device, sizes[i], sizes[i]);
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
  std::vector<std::shared_ptr<ImageBuffer>> images;

  for (size_t i = 0; i < sizes.size(); ++i)
  {
    filters.push_back(device.newFilter("RT"));
    REQUIRE(bool(filters[i]));

    images.push_back(makeConstImage(device, sizes[i], sizes[i]));
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
  DeviceRef device = makeDevice();
  REQUIRE(bool(device));
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

// -------------------------------------------------------------------------------------------------

TEST_CASE("multiple devices", "[multi_device]")
{
  const std::vector<int> sizes = {111, 256, 80};

  std::vector<DeviceRef> devices;
  std::vector<FilterRef> filters;
  std::vector<std::shared_ptr<ImageBuffer>> images;

  for (size_t i = 0; i < sizes.size(); ++i)
  {
    devices.push_back(makeDevice());
    REQUIRE(devices[i]);
    devices[i].commit();
    REQUIRE(devices[i].getError() == Error::None);

    filters.push_back(devices[i].newFilter("RT"));
    REQUIRE(bool(filters[i]));

    images.push_back(makeConstImage(devices[i], sizes[i], sizes[i]));
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

// -------------------------------------------------------------------------------------------------

TEST_CASE("image buffers", "[image_buffers]")
{
  const int W = 198;
  const int H = 300;

  DeviceRef device = makeDevice();
  REQUIRE(bool(device));
  device.commit();
  REQUIRE(device.getError() == Error::None);

  FilterRef filter = device.newFilter("RT");
  REQUIRE(bool(filter));

  auto color  = makeConstImage(device, W, H);
  auto output = makeImage(device, W, H);
  setFilterImage(filter, "color",  color,  true);
  setFilterImage(filter, "output", output, true);

  filter.commit();
  REQUIRE(device.getError() == Error::None);

  filter.execute();
  REQUIRE(device.getError() == Error::None);
}

// -------------------------------------------------------------------------------------------------

TEST_CASE("filter update", "[filter_update]")
{
  const int W = 211;
  const int H = 599;

  DeviceRef device = makeDevice();
  REQUIRE(bool(device));
  device.commit();
  REQUIRE(device.getError() == Error::None);

  FilterRef filter = device.newFilter("RT");
  REQUIRE(bool(filter));

  auto color  = makeConstImage(device, W, H);
  auto albedo = makeConstImage(device, W, H);
  auto output = makeConstImage(device, W, H);
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
    color = makeConstImage(device, W, H);
    setFilterImage(filter, "color", color);
  }

  SECTION("filter update: different image size")
  {
    color  = makeConstImage(device, W*2, H*2);
    albedo = makeConstImage(device, W*2, H*2);
    output = makeConstImage(device, W*2, H*2);
    setFilterImage(filter, "color",  color);
    setFilterImage(filter, "albedo", albedo);
    setFilterImage(filter, "output", output);
  }

  SECTION("filter update: unset image")
  {
    filter.unsetImage("albedo");
  }

  SECTION("filter update: unset image by setting to null")
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

// -------------------------------------------------------------------------------------------------

TEST_CASE("async filter", "[async_filter]")
{
  const int W = 799;
  const int H = 601;

  DeviceRef device = makeDevice();
  REQUIRE(bool(device));
  device.commit();
  REQUIRE(device.getError() == Error::None);

  FilterRef filter = device.newFilter("RT");
  REQUIRE(bool(filter));

  auto color  = makeConstImage(device, W, H);
  auto albedo = makeConstImage(device, W, H);
  auto output = makeImage(device, W, H);

  setFilterImage(filter, "color",  color);
  setFilterImage(filter, "output", output);
  filter.set("hdr", true);

  filter.commit();
  REQUIRE(device.getError() == Error::None);

  for (int i = 0; i < 3; ++i)
    filter.executeAsync();

  setFilterImage(filter, "albedo", albedo);
  filter.set("hdr", false);

  filter.commit();
  REQUIRE(device.getError() == Error::None);

  for (int i = 0; i < 2; ++i)
    filter.executeAsync();

  device.sync();
  REQUIRE(device.getError() == Error::None);

  filter.executeAsync();
  filter = nullptr;

  REQUIRE(device.getError() == Error::None);
}

// -------------------------------------------------------------------------------------------------

void imageSizeTest(DeviceRef& device, int W, int H, bool execute = true)
{
  FilterRef filter = device.newFilter("RT");
  REQUIRE(bool(filter));

  auto color  = makeConstImage(device, W, H);
  auto output = makeImage(device, W, H);
  setFilterImage(filter, "color",  color);
  setFilterImage(filter, "output", output);

  filter.commit();
  REQUIRE(device.getError() == Error::None);

  if (execute)
  {
    filter.execute();
    REQUIRE(device.getError() == Error::None);
  }
}

TEST_CASE("image size", "[size]")
{
  DeviceRef device = makeDevice();
  REQUIRE(bool(device));
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

  SECTION("image size: 8192x4320")
  {
    imageSizeTest(device, 8192, 4320, false);
  }
}

// -------------------------------------------------------------------------------------------------

void sanitizationTest(DeviceRef& device, bool hdr, float value)
{
  const int W = 191;
  const int H = 347;

  FilterRef filter = device.newFilter("RT");
  REQUIRE(bool(filter));

  auto input  = makeConstImage(device, W, H, 3, value);
  auto output = makeImage(device, W, H, 3);
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
  DeviceRef device = makeDevice();
  REQUIRE(bool(device));
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

// -------------------------------------------------------------------------------------------------

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
  REQUIRE(n >= progress->n); // n must not decrease
  progress->n = n;
  return n < progress->nMax; // cancel if reached nMax
}

void progressTest(DeviceRef& device, double nMax = 1000)
{
  const int W = 1283;
  const int H = 727;

  FilterRef filter = device.newFilter("RT");
  REQUIRE(bool(filter));

  auto image = makeConstImage(device, W, H);
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
    // Execution should be cancelled but it's not guaranteed
    Error error = device.getError();
    REQUIRE((error == Error::None || error == Error::Cancelled));
    // Check whether the callback has not been called after requesting cancellation
    REQUIRE(progress.n >= nMax);
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
  DeviceRef device = makeDevice();
  REQUIRE(bool(device));
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

int main(int argc, char* argv[])
{
  Catch::Session session;

  std::string deviceStr = "default";

  using namespace Catch::clara;
  auto cli
    = session.cli()
    | Opt(deviceStr, "[0-9]+|default|cpu|sycl|cuda|hip")
        ["--device"]
        ("Open Image Denoise device to use");

  session.cli(cli);

  int returnCode = session.applyCommandLine(argc, argv);
  if (returnCode != 0)
    return returnCode;

  try
  {
    if (isdigit(deviceStr[0]))
      physicalDevice = fromString<int>(deviceStr);
    else
      deviceType = fromString<DeviceType>(deviceStr);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return session.run();
}