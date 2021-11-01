// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <cassert>
#include <cmath>
#include <regex>
#include <chrono>
#include <thread>

#ifdef VTUNE
#include <ittnotify.h>
#endif

#include <OpenImageDenoise/oidn.hpp>

#include "common/math.h"
#include "common/timer.h"
#include "apps/utils/image_io.h"
#include "apps/utils/random.h"
#include "apps/utils/arg_parser.h"

OIDN_NAMESPACE_USING
using namespace oidn;

int width  = -1;
int height = -1;
Format dataType = Format::Float;
int numRuns = 4;
int maxMemoryMB = -1;
bool inplace = false;

void printUsage()
{
  std::cout << "Intel(R) Open Image Denoise - Benchmark" << std::endl;
  std::cout << "usage: oidnBenchmark [-d/--device default|cpu]" << std::endl
            << "                     [-r/--run regex] [-n times]" << std::endl
            << "                     [-s/--size width height]" << std::endl
            << "                     [-t/--type float|half]" << std::endl
            << "                     [--threads n] [--affinity 0|1] [--maxmem MB] [--inplace]" << std::endl
            << "                     [-v/--verbose 0-3]" << std::endl
            << "                     [-l/--list] [-h/--help]" << std::endl;
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

// Initializes an image with random values
void initImage(ImageBuffer& image, Random& rng, float minValue, float maxValue)
{
  for (size_t i = 0; i < image.size(); ++i)
    image.set(i, minValue + rng.get1f() * (maxValue - minValue));
}

// Runs a benchmark
void runBenchmark(DeviceRef& device, const Benchmark& bench)
{
  std::cout << bench.name << " ..." << std::flush;

  // Initialize the filter and the buffers
  FilterRef filter = device.newFilter(bench.filter.c_str());
  Random rng;

  std::shared_ptr<ImageBuffer> input;

  std::shared_ptr<ImageBuffer> albedo;
  if (bench.hasInput("alb"))
  {
    input = albedo = std::make_shared<ImageBuffer>(device, bench.width, bench.height, 3, dataType);
    initImage(*albedo, rng, 0.f, 1.f);
    filter.setImage("albedo", albedo->data(), albedo->format(), bench.width, bench.height);
  }

  std::shared_ptr<ImageBuffer> normal;
  if (bench.hasInput("nrm"))
  {
    input = normal = std::make_shared<ImageBuffer>(device, bench.width, bench.height, 3, dataType);
    initImage(*normal, rng, -1.f, 1.f);
    filter.setImage("normal", normal->data(), normal->format(), bench.width, bench.height);
  }

  std::shared_ptr<ImageBuffer> color;
  if (bench.hasInput("hdr"))
  {
    input = color = std::make_shared<ImageBuffer>(device, bench.width, bench.height, 3, dataType);
    initImage(*color, rng, 0.f, 100.f);
    filter.setImage("color", color->data(), color->format(), bench.width, bench.height);
    filter.set("hdr", true);
  }
  else if (bench.hasInput("ldr"))
  {
    input = color = std::make_shared<ImageBuffer>(device, bench.width, bench.height, 3, dataType);
    initImage(*color, rng, 0.f, 1.f);
    filter.setImage("color", color->data(), color->format(), bench.width, bench.height);
    filter.set("hdr", false);
  }

  std::shared_ptr<ImageBuffer> output;
  if (inplace)
    output = input;
  else
    output = std::make_shared<ImageBuffer>(device, bench.width, bench.height, 3, dataType);
  filter.setImage("output", output->data(), output->format(), bench.width, bench.height);

  if (maxMemoryMB >= 0)
    filter.set("maxMemoryMB", maxMemoryMB);

  filter.commit();

  // Warmup loop
  const int numBenchmarkRuns = max(numRuns - 1, 1);
  const int numWarmupRuns = numRuns - numBenchmarkRuns;
  for (int i = 0; i < numWarmupRuns; ++i)
    filter.execute();

  // Benchmark loop
  Timer timer;

  #ifdef VTUNE
    __itt_resume();
  #endif

  for (int i = 0; i < numBenchmarkRuns; ++i)
    filter.execute();

  #ifdef VTUNE
    __itt_pause();
  #endif

  // Print results
  const double totalTime = timer.query();
  const double avgTime = totalTime / numBenchmarkRuns;
  std::cout << " " << avgTime * 1000. << " msec/image" << std::endl;

  // Cooldown
  const int sleepTime = int(std::ceil(totalTime / 2.));
  std::this_thread::sleep_for(std::chrono::seconds(sleepTime));
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
        const auto val = args.getNextValue();
        if (val == "default" || val == "Default")
          deviceType = DeviceType::Default;
        else if (val == "cpu" || val == "CPU")
          deviceType = DeviceType::CPU;
        else
          throw std::invalid_argument("invalid device");
      }
      else if (opt == "r" || opt == "run")
        run = args.getNextValue();
      else if (opt == "n")
      {
        numRuns = args.getNextValueInt();
        if (numRuns <= 0)
          throw std::runtime_error("invalid number of runs");
      }
      else if (opt == "s" || opt == "size")
      {
        width  = args.getNextValueInt();
        height = args.getNextValueInt();
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
        numThreads = args.getNextValueInt();
      else if (opt == "affinity")
        setAffinity = args.getNextValueInt();
      else if (opt == "maxmem" || opt == "maxMemoryMB")
        maxMemoryMB = args.getNextValueInt();
      else if (opt == "inplace")
        inplace = true;
      else if (opt == "v" || opt == "verbose")
        verbose = args.getNextValueInt();
      else if (opt == "l" || opt == "list")
        run = "";
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

  #if defined(OIDN_X64)
    // Enable the FTZ and DAZ flags to maximize performance
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  #endif

    // Initialize the device
    DeviceRef device = newDevice(deviceType);

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
    for (const auto& bench : benchmarks)
    {
      if (std::regex_match(bench.name, runExpr))
        runBenchmark(device, bench);
    }
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
