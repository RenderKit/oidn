// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common/common.h"
#include "common/timer.h"
#include "utils/arg_parser.h"
#include "utils/image_buffer.h"
#include "utils/device_info.h"
#include "utils/random.h"
#include <iostream>
#include <cassert>
#include <cmath>
#include <regex>
#include <chrono>
#include <thread>
#ifdef VTUNE
#include <ittnotify.h>
#endif

OIDN_NAMESPACE_USING

int width  = -1;
int height = -1;
DataType dataType = DataType::Float32;
Quality quality = Quality::Default;
Storage bufferStorage = Storage::Device; // maximize performance by default
bool bufferCopy = false; // include copying between host and device in the measurements
int numRuns = 0;
int maxMemoryMB = -1;
bool inplace = false;

void printUsage()
{
  std::cout << "Intel(R) Open Image Denoise - Benchmark" << std::endl;
  std::cout << "usage: oidnBenchmark [-d/--device [0-9]+|default|cpu|sycl|cuda|hip|metal]" << std::endl
            << "                     [-r/--run regex] [-n times_to_run]" << std::endl
            << "                     [-s/--size width height]" << std::endl
            << "                     [-t/--type float|half]" << std::endl
            << "                     [-q/--quality default|h|high|b|balanced|f|fast]" << std::endl
            << "                     [--threads n] [--affinity 0|1] [--maxmem MB] [--inplace]" << std::endl
            << "                     [--buffer host(copy)|device(copy)|managed(copy)]" << std::endl
            << "                     [-v/--verbose 0-3]" << std::endl
            << "                     [--ld|--list_devices] [-l/--list] [-h/--help]" << std::endl;
}

void errorCallback(void* userPtr, Error error, const char* message)
{
  throw std::runtime_error(message);
}

// Benchmark descriptor
struct Benchmark
{
  std::string name;
  std::string filter;
  std::vector<std::string> inputs;
  int width;
  int height;

  bool hasInput(const std::string& input) const
  {
    return std::find(inputs.begin(), inputs.end(), input) != inputs.end();
  }
};

// List of all benchmarks
std::vector<Benchmark> benchmarks;

// Adds a benchmark to the list
void addBenchmark(const std::string& filter, const std::vector<std::string>& inputs, const std::pair<int, int>& size)
{
  Benchmark bench;
  bench.name = filter;
  bench.name += ".";
  for (size_t i = 0; i < inputs.size(); ++i)
  {
    if (i > 0) bench.name += "_";
    bench.name += inputs[i];
  }
  bench.name += "." + toString(size.first) + "x" + toString(size.second);

  bench.filter = filter;
  bench.inputs = inputs;
  bench.width  = size.first;
  bench.height = size.second;

  benchmarks.push_back(bench);
}

std::shared_ptr<ImageBuffer> newImage(DeviceRef& device, int width, int height)
{
  return std::make_shared<ImageBuffer>(device, width, height, 3, dataType, bufferStorage, bufferCopy);
}

// Initializes an image with random values
void initImage(ImageBuffer& image, Random& rng, float minValue, float maxValue)
{
  for (size_t i = 0; i < image.getSize(); ++i)
    image.set(i, minValue + rng.getFloat() * (maxValue - minValue));
  image.toDevice();
}

// Runs a benchmark and returns the total runtime
double runBenchmark(DeviceRef& device, const Benchmark& bench)
{
  std::cout << bench.name << " ..." << std::flush;

  // Initialize the filter and the buffers
  FilterRef filter = device.newFilter(bench.filter.c_str());
  Random rng;

  std::shared_ptr<ImageBuffer> input;

  std::shared_ptr<ImageBuffer> albedo;
  if (bench.hasInput("alb") || bench.hasInput("calb"))
  {
    input = albedo = newImage(device, bench.width, bench.height);
    initImage(*albedo, rng, 0.f, 1.f);
    filter.setImage("albedo", albedo->getBuffer(), albedo->getFormat(), bench.width, bench.height);
  }

  std::shared_ptr<ImageBuffer> normal;
  if (bench.hasInput("nrm") || bench.hasInput("cnrm"))
  {
    input = normal = newImage(device, bench.width, bench.height);
    initImage(*normal, rng, -1.f, 1.f);
    filter.setImage("normal", normal->getBuffer(), normal->getFormat(), bench.width, bench.height);
  }

  std::shared_ptr<ImageBuffer> color;
  if (bench.hasInput("hdr"))
  {
    input = color = newImage(device, bench.width, bench.height);
    initImage(*color, rng, 0.f, 100.f);
    filter.setImage("color", color->getBuffer(), color->getFormat(), bench.width, bench.height);
    if (bench.filter != "RTLightmap")
      filter.set("hdr", true);
  }
  else if (bench.hasInput("ldr"))
  {
    input = color = newImage(device, bench.width, bench.height);
    initImage(*color, rng, 0.f, 1.f);
    filter.setImage("color", color->getBuffer(), color->getFormat(), bench.width, bench.height);
    filter.set("hdr", false);
  }

  if (bench.hasInput("calb") || bench.hasInput("cnrm"))
    filter.set("cleanAux", true);

  std::shared_ptr<ImageBuffer> output;
  if (inplace)
    output = input;
  else
    output = newImage(device, bench.width, bench.height);
  filter.setImage("output", output->getBuffer(), output->getFormat(), bench.width, bench.height);

  if (quality != Quality::Default)
    filter.set("quality", quality);

  if (maxMemoryMB >= 0)
    filter.set("maxMemoryMB", maxMemoryMB);

  filter.commit();

  auto executeFilterAsync = [&]()
  {
    if (bufferCopy)
    {
      input->toDeviceAsync();
      if (albedo)
        albedo->toDeviceAsync();
      if (normal)
        normal->toDeviceAsync();
    }

    filter.executeAsync();

    if (bufferCopy)
      output->toHostAsync();
  };

  // Warmup / determine number of benchmark runs
  int numBenchmarkRuns = 0;
  if (numRuns > 0)
  {
    numBenchmarkRuns = std::max(numRuns - 1, 1);
    const int numWarmupRuns = numRuns - numBenchmarkRuns;
    for (int i = 0; i < numWarmupRuns; ++i)
      executeFilterAsync();
    device.sync();
  }
  else
  {
    // First warmup run
    executeFilterAsync();
    device.sync();

    // Second warmup run, measure time
    Timer timer;
    executeFilterAsync();
    device.sync();
    double warmupTime = timer.query();

    // Benchmark for at least 0.5 seconds or 3 times
    numBenchmarkRuns = std::max(int(0.5 / warmupTime), 3);
  }

  // Benchmark loop
  Timer timer;
  Timer asyncTimer;
  double totalAsyncTime = 0;

  #ifdef VTUNE
    __itt_resume();
  #endif

  for (int i = 0; i < numBenchmarkRuns; ++i)
  {
    asyncTimer.reset();
    executeFilterAsync();
    totalAsyncTime += asyncTimer.query();
    device.sync();
  }

  #ifdef VTUNE
    __itt_pause();
  #endif

  // Print results
  const double totalTime = timer.query();
  const double avgTime = totalTime / numBenchmarkRuns;
  const double avgAsyncTime = totalAsyncTime / numBenchmarkRuns;
  std::cout << " " << avgTime * 1000 << " msec/image"
            << " (host " << avgAsyncTime * 1000 << " msec/image)"
            << std::endl;

  return totalTime;
}

// Adds all benchmarks to the list
void addAllBenchmarks()
{
  std::vector<std::pair<int, int>> sizes;

  // Filter: RT
#if defined(OIDN_FILTER_RT)
  if (width < 0)
    sizes = {{1920, 1080}, {3840, 2160}, {1280, 720}};
  else
    sizes = {{width, height}};

  for (const auto& size : sizes)
  {
    addBenchmark("RT", {"hdr", "alb", "nrm"}, size);
    addBenchmark("RT", {"ldr", "alb", "nrm"}, size);
    addBenchmark("RT", {"hdr", "calb", "cnrm"}, size);
    addBenchmark("RT", {"ldr", "calb", "cnrm"}, size);
  }
#endif

  // Filter: RTLightmap
#if defined(OIDN_FILTER_RTLIGHTMAP)
  if (width < 0)
    sizes = {{2048, 2048}, {4096, 4096}, {1024, 1024}};

  for (const auto& size : sizes)
  {
    addBenchmark("RTLightmap", {"hdr"}, size);
  }
#endif
}

int main(int argc, char* argv[])
{
  DeviceType deviceType = DeviceType::Default;
  PhysicalDeviceRef physicalDevice;
  std::string run = ".*";
  int numThreads = -1;
  int setAffinity = -1;
  int verbose = -1;

  try
  {
    ArgParser args(argc, argv);
    while (args.hasNext())
    {
      std::string opt = args.getNextOpt();
      if (opt == "d" || opt == "dev" || opt == "device")
      {
        std::string value = args.getNext();
        if (isdigit(value[0]))
          physicalDevice = fromString<int>(value);
        else
          deviceType = fromString<DeviceType>(value);
      }
      else if (opt == "r" || opt == "run")
        run = args.getNextValue();
      else if (opt == "n")
      {
        numRuns = args.getNextValue<int>();
        if (numRuns <= 0)
          throw std::runtime_error("invalid number of runs");
      }
      else if (opt == "s" || opt == "size")
      {
        width  = args.getNextValue<int>();
        height = args.getNextValue<int>();
        if (width < 1 || height < 1)
          throw std::runtime_error("invalid image size");
      }
      else if (opt == "t" || opt == "type")
      {
        const auto val = toLower(args.getNextValue());
        if (val == "f" || val == "float" || val == "fp32")
          dataType = DataType::Float32;
        else if (val == "h" || val == "half" || val == "fp16")
          dataType = DataType::Float16;
        else
          throw std::runtime_error("invalid data type");
      }
      else if (opt == "q" || opt == "quality")
      {
        const auto val = toLower(args.getNextValue());
        if (val == "default")
          quality = Quality::Default;
        else if (val == "h" || val == "high")
          quality = Quality::High;
        else if (val == "b" || val == "balanced")
          quality = Quality::Balanced;
        else if (val == "f" || val == "fast")
          quality = Quality::Fast;
        else
          throw std::runtime_error("invalid filter quality mode");
      }
      else if (opt == "threads")
        numThreads = args.getNextValue<int>();
      else if (opt == "affinity")
        setAffinity = args.getNextValue<int>();
      else if (opt == "maxmem" || opt == "maxMemoryMB")
        maxMemoryMB = args.getNextValue<int>();
      else if (opt == "inplace")
        inplace = true;
      else if (opt == "buffer")
      {
        const auto val = toLower(args.getNextValue());
        if (val == "host")
          bufferStorage = Storage::Host;
        else if (val == "hostcopy")
          std::tie(bufferStorage, bufferCopy) = std::make_pair(Storage::Host, true);
        else if (val == "device")
          bufferStorage = Storage::Device;
        else if (val == "devicecopy")
          std::tie(bufferStorage, bufferCopy) = std::make_pair(Storage::Device, true);
        else if (val == "managed")
          bufferStorage = Storage::Managed;
        else if (val == "managedcopy")
          std::tie(bufferStorage, bufferCopy) = std::make_pair(Storage::Managed, true);
        else
          throw std::runtime_error("invalid storage mode");
      }
      else if (opt == "v" || opt == "verbose")
        verbose = args.getNextValue<int>();
      else if (opt == "l" || opt == "list")
        run = "";
      else if (opt == "ld" || opt == "list_devices" || opt == "list-devices" || opt == "listDevices" || opt == "listdevices")
        return printPhysicalDevices();
      else if (opt == "h" || opt == "help")
      {
        printUsage();
        return 1;
      }
      else
        throw std::invalid_argument("invalid argument: '" + opt + "'");
    }

    // Add the benchmarks to the list
    addAllBenchmarks();

    if (run.empty())
    {
      // List all benchmarks
      for (const auto& bench : benchmarks)
        std::cout << bench.name << std::endl;
      return 0;
    }

  #if defined(OIDN_ARCH_X64)
    // Enable the FTZ and DAZ flags to maximize performance
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  #endif

    // Initialize the device
    DeviceRef device;
    if (physicalDevice)
      device = physicalDevice.newDevice();
    else
      device = newDevice(deviceType);

    if (verbose >= 0)
      device.set("verbose", verbose);

    const char* errorMessage;
    if (device.getError(errorMessage) != Error::None)
      throw std::runtime_error(errorMessage);
    device.setErrorFunction(errorCallback);

    if (numThreads > 0)
      device.set("numThreads", numThreads);
    if (setAffinity >= 0)
      device.set("setAffinity", bool(setAffinity));

    device.commit();

    if (bufferStorage == Storage::Managed && !device.get<bool>("managedMemorySupported"))
      throw std::runtime_error("managed memory is not supported by the device");

    // Run the benchmarks
    const auto runExpr = std::regex(run);
    double prevBenchTime = 0;

    for (const auto& bench : benchmarks)
    {
      if (std::regex_match(bench.name, runExpr))
      {
        // Cooldown
        if (prevBenchTime > 0)
        {
          const int sleepTime = int(std::ceil(prevBenchTime / 2.));
          std::this_thread::sleep_for(std::chrono::seconds(sleepTime));
        }

        prevBenchTime = runBenchmark(device, bench);
      }
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
