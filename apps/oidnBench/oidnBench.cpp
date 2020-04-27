// Copyright 2009-2020 Intel Corporation
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

using namespace oidn;

int numRuns = -1;
int maxMemoryMB = -1;

void printUsage()
{
  std::cout << "Intel(R) Open Image Denoise - Benchmark" << std::endl;
  std::cout << "usage: oidnBench [-r/--run regex] [-n times]" << std::endl
            << "                 [--threads n] [--affinity 0|1] [--maxmem MB]" << std::endl
            << "                 [-v/--verbose 0-3]" << std::endl
            << "                 [-l/--list] [-h/--help]" << std::endl;
}

void errorCallback(void* userPtr, oidn::Error error, const char* message)
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
  for (int i = 0; i < image.getSize(); ++i)
    image[i] = minValue + rng.get1f() * (maxValue - minValue);
}

// Runs a benchmark
void runBenchmark(oidn::DeviceRef& device, const Benchmark& bench)
{
  std::cout << bench.name << " ..." << std::flush;

  // Initialize the filter and the buffers
  oidn::FilterRef filter = device.newFilter(bench.filter.c_str());
  Random rng;

  ImageBuffer color;
  if (bench.hasInput("hdr"))
  {
    color = ImageBuffer(bench.width, bench.height, 3);
    initImage(color, rng, 0.f, 100.f);
    filter.setImage("color", color.getData(), oidn::Format::Float3, bench.width, bench.height);
    filter.set("hdr", true);
  }
  else if (bench.hasInput("ldr"))
  {
    color = ImageBuffer(bench.width, bench.height, 3);
    initImage(color, rng, 0.f, 1.f);
    filter.setImage("color", color.getData(), oidn::Format::Float3, bench.width, bench.height);
    filter.set("hdr", false);
  }

  ImageBuffer albedo;
  if (bench.hasInput("alb"))
  {
    albedo = ImageBuffer(bench.width, bench.height, 3);
    initImage(albedo, rng, 0.f, 1.f);
    filter.setImage("albedo", albedo.getData(), oidn::Format::Float3, bench.width, bench.height);
  }

  ImageBuffer normal;
  if (bench.hasInput("nrm"))
  {
    normal = ImageBuffer(bench.width, bench.height, 3);
    initImage(normal, rng, -1.f, 1.f);
    filter.setImage("normal", normal.getData(), oidn::Format::Float3, bench.width, bench.height);
  }

  ImageBuffer output(bench.width, bench.height, 3);
  filter.setImage("output", output.getData(), oidn::Format::Float3, bench.width, bench.height);

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
  const std::vector<std::pair<int, int>> sizes =
  {
    {1920, 1080},
    {3840, 2160},
    {1280, 720},
  };

  for (const auto& size : sizes)
  {
    addBenchmark("RT", {"hdr", "alb", "nrm"}, size);
    addBenchmark("RT", {"ldr", "alb", "nrm"}, size);
  }

  const std::vector<std::pair<int, int>> lightmapSizes =
  {
    {2048, 2048},
    {4096, 4096},
    {1024, 1024},
  };

  for (const auto& size : lightmapSizes)
  {
    addBenchmark("RTLightmap", {"hdr"}, size);
  }
}

int main(int argc, char* argv[])
{
  std::string run = ".*";
  int numThreads = -1;
  int setAffinity = -1;
  int verbose = -1;

  // Add the benchmarks to the list
  addAllBenchmarks();

  try
  {
    ArgParser args(argc, argv);
    while (args.hasNext())
    {
      std::string opt = args.getNextOpt();
      if (opt == "r" || opt == "run")
        run = args.getNextValue();
      else if (opt == "n")
      {
        numRuns = args.getNextValueInt();
        if (numRuns <= 0)
          throw std::runtime_error("invalid number of runs");
      }
      else if (opt == "threads")
        numThreads = args.getNextValueInt();
      else if (opt == "affinity")
        setAffinity = args.getNextValueInt();
      else if (opt == "maxmem")
        maxMemoryMB = args.getNextValueInt();
      else if (opt == "v" || opt == "verbose")
        verbose = args.getNextValueInt();
      else if (opt == "l" || opt == "list")
      {
        // List all benchmarks
        for (const auto& bench : benchmarks)
          std::cout << bench.name << std::endl;
        return 0;
      }
      else if (opt == "h" || opt == "help")
      {
        printUsage();
        return 1;
      }
      else
        throw std::invalid_argument("invalid argument");
    }

    // Enable the FTZ and DAZ flags to maximize performance
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    // Initialize the device
    oidn::DeviceRef device = oidn::newDevice();

    const char* errorMessage;
    if (device.getError(errorMessage) != oidn::Error::None)
      throw std::runtime_error(errorMessage);
    device.setErrorFunction(errorCallback);

    if (numThreads > 0)
      device.set("numThreads", numThreads);
    if (setAffinity >= 0)
      device.set("setAffinity", bool(setAffinity));
    if (verbose >= 0)
      device.set("verbose", verbose);
    device.commit();

    // Run the benchmarks
    if (numRuns < 0)
       numRuns = device.get<int>("numThreads");
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
