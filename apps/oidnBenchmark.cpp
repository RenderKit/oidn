// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common/common.h"
#include "utils/arg_parser.h"
#include "utils/image_buffer.h"
#include "utils/device_info.h"
#include "utils/random.h"
#include "utils/timer.h"
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
Format dataType = Format::Float;
int numRuns = 0;
int maxMemoryMB = -1;
bool inplace = false;

void printUsage()
{
  std::cout << "Intel(R) Open Image Denoise - Benchmark" << std::endl;
  std::cout << "usage: oidnBenchmark [-d/--device [0-9]+|default|cpu|sycl|cuda|hip]" << std::endl
            << "                     [-r/--run regex] [-n times_to_run]" << std::endl
            << "                     [-s/--size width height]" << std::endl
            << "                     [-t/--type float|half]" << std::endl
            << "                     [--threads n] [--affinity 0|1] [--maxmem MB] [--inplace]" << std::endl
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
  // Create a buffer stored on the device to maximize performance
  return std::make_shared<ImageBuffer>(device, width, height, 3, dataType, Storage::Device);
}

// Initializes an image with random values
void initImage(ImageBuffer& image, Random& rng, float minValue, float maxValue)
{
  image.map(Access::WriteDiscard);
  for (size_t i = 0; i < image.getSize(); ++i)
    image.set(i, minValue + rng.getFloat() * (maxValue - minValue));
  image.unmap();
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
  if (bench.hasInput("alb"))
  {
    input = albedo = newImage(device, bench.width, bench.height);
    initImage(*albedo, rng, 0.f, 1.f);
    filter.setImage("albedo", albedo->getBuffer(), albedo->getFormat(), bench.width, bench.height);
  }

  std::shared_ptr<ImageBuffer> normal;
  if (bench.hasInput("nrm"))
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
    filter.set("hdr", true);
  }
  else if (bench.hasInput("ldr"))
  {
    input = color = newImage(device, bench.width, bench.height);
    initImage(*color, rng, 0.f, 1.f);
    filter.setImage("color", color->getBuffer(), color->getFormat(), bench.width, bench.height);
    filter.set("hdr", false);
  }

  std::shared_ptr<ImageBuffer> output;
  if (inplace)
    output = input;
  else
    output = newImage(device, bench.width, bench.height);
  filter.setImage("output", output->getBuffer(), output->getFormat(), bench.width, bench.height);

  if (maxMemoryMB >= 0)
    filter.set("maxMemoryMB", maxMemoryMB);

  filter.commit();

  // Warmup / determine number of benchmark runs
  int numBenchmarkRuns = 0;
  if (numRuns > 0)
  {
    numBenchmarkRuns = std::max(numRuns - 1, 1);
    const int numWarmupRuns = numRuns - numBenchmarkRuns;
    for (int i = 0; i < numWarmupRuns; ++i)
      filter.execute();
  }
  else
  {
    // First warmup run
    filter.execute();

    // Second warmup run, measure time
    Timer timer;
    filter.execute();
    double warmupTime = timer.query();

    // Benchmark for at least 0.5 seconds or 3 times
    numBenchmarkRuns = std::max(int(0.5 / warmupTime), 3);
  }

  // Benchmark loop
  Timer timer;

  #ifdef VTUNE
    __itt_resume();
  #endif

  for (int i = 0; i < numBenchmarkRuns; ++i)
    filter.executeAsync();

  device.sync();

  #ifdef VTUNE
    __itt_pause();
  #endif

  // Print results
  const double totalTime = timer.query();
  const double avgTime = totalTime / numBenchmarkRuns;
  std::cout << " " << avgTime * 1000. << " msec/image" << std::endl;

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
        const auto val = args.getNextValue();
        if (val == "f" || val == "float" || val == "Float" || val == "fp32")
          dataType = Format::Float;
        else if (val == "h" || val == "half" || val == "Half" || val == "fp16")
          dataType = Format::Half;
        else
          throw std::runtime_error("invalid data type");
      }
      else if (opt == "threads")
        numThreads = args.getNextValue<int>();
      else if (opt == "affinity")
        setAffinity = args.getNextValue<int>();
      else if (opt == "maxmem" || opt == "maxMemoryMB")
        maxMemoryMB = args.getNextValue<int>();
      else if (opt == "inplace")
        inplace = true;
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
        throw std::invalid_argument("invalid argument");
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
