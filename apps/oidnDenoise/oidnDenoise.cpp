// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <fstream>
#include <cassert>
#include <cmath>
#include <signal.h>

#ifdef VTUNE
#include <ittnotify.h>
#endif

#include <OpenImageDenoise/oidn.hpp>

#include "common/timer.h"
#include "apps/utils/image_io.h"
#include "apps/utils/arg_parser.h"

using namespace oidn;

void printUsage()
{
  std::cout << "Intel(R) Open Image Denoise - Example" << std::endl;
  std::cout << "usage: oidnDenoise [-f/--filter RT|RTLightmap]" << std::endl
            << "                   [--ldr color.pfm] [--srgb] [--hdr color.pfm]" << std::endl
            << "                   [--alb albedo.pfm] [--nrm normal.pfm]" << std::endl
            << "                   [-o/--output output.pfm] [-r/--ref reference_output.pfm]" << std::endl
            << "                   [-w/--weights weights.tza]" << std::endl
            << "                   [--threads n] [--affinity 0|1] [--maxmem MB]" << std::endl
            << "                   [--bench ntimes] [-v/--verbose 0-3]" << std::endl
            << "                   [-h/--help]" << std::endl;
}

void errorCallback(void* userPtr, oidn::Error error, const char* message)
{
  throw std::runtime_error(message);
}

volatile bool isCancelled = false;

void signalHandler(int signal)
{
  isCancelled = true;
}

bool progressCallback(void* userPtr, double n)
{
  if (isCancelled)
    return false;
  std::cout << "\rDenoising " << int(n * 100.) << "%" << std::flush;
  return true;
}

std::vector<char> loadFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::binary);
  if (file.fail())
    throw std::runtime_error("cannot open file: " + filename);
  file.seekg(0, file.end);
  const size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  if (file.fail())
    throw std::runtime_error("error reading from file");
  return buffer;
}

int main(int argc, char* argv[])
{
  std::string filterType = "RT";
  std::string colorFilename, albedoFilename, normalFilename;
  std::string outputFilename, refFilename;
  std::string weightsFilename;
  bool hdr = false;
  bool srgb = false;
  int numBenchmarkRuns = 0;
  int numThreads = -1;
  int setAffinity = -1;
  int maxMemoryMB = -1;
  int verbose = -1;

  // Parse the arguments
  if (argc == 1)
  {
    printUsage();
    return 1;
  }

  try
  {
    ArgParser args(argc, argv);
    while (args.hasNext())
    {
      std::string opt = args.getNextOpt();
      if (opt == "f" || opt == "filter")
        filterType = args.getNextValue();
      else if (opt == "ldr")
      {
        colorFilename = args.getNextValue();
        hdr = false;
      }
      else if (opt == "hdr")
      {
        colorFilename = args.getNextValue();
        hdr = true;
      }
      else if (opt == "srgb")
        srgb = true;
      else if (opt == "alb" || opt == "albedo")
        albedoFilename = args.getNextValue();
      else if (opt == "nrm" || opt == "normal")
        normalFilename = args.getNextValue();
      else if (opt == "o" || opt == "out" || opt == "output")
        outputFilename = args.getNextValue();
      else if (opt == "r" || opt == "ref" || opt == "reference")
        refFilename = args.getNextValue();
      else if (opt == "w" || opt == "weights")
        weightsFilename = args.getNextValue();
      else if (opt == "bench" || opt == "benchmark")
        numBenchmarkRuns = std::max(args.getNextValueInt(), 0);
      else if (opt == "threads")
        numThreads = args.getNextValueInt();
      else if (opt == "affinity")
        setAffinity = args.getNextValueInt();
      else if (opt == "maxmem")
        maxMemoryMB = args.getNextValueInt();
      else if (opt == "v" || opt == "verbose")
        verbose = args.getNextValueInt();
      else if (opt == "h" || opt == "help")
      {
        printUsage();
        return 1;
      }
      else
        throw std::invalid_argument("invalid argument");
    }

    if (colorFilename.empty())
      throw std::runtime_error("no color image specified");

    if (!refFilename.empty() && numBenchmarkRuns > 0)
      throw std::runtime_error("reference and benchmark modes cannot be enabled at the same time");

    // Set MXCSR flags
    if (!refFilename.empty())
    {
      // In reference mode we have to disable the FTZ and DAZ flags to get accurate results
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    }
    else
    {
      // Enable the FTZ and DAZ flags to maximize performance
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }

    // Load the input image
    ImageBuffer color, albedo, normal;
    ImageBuffer ref;

    std::cout << "Loading input" << std::endl;

    color = loadImage(colorFilename, 3, srgb);

    if (!albedoFilename.empty())
    {
      albedo = loadImage(albedoFilename, 3, false);
      if (albedo.getDims() != color.getDims())
        throw std::runtime_error("invalid albedo image");
    }

    if (!normalFilename.empty())
    {
      normal = loadImage(normalFilename, 3);
      if (normal.getDims() != color.getDims())
        throw std::runtime_error("invalid normal image");
    }

    if (!refFilename.empty())
    {
      ref = loadImage(refFilename, 3, srgb);
      if (ref.getDims() != color.getDims())
        throw std::runtime_error("invalid reference output image");
    }

    const int width  = color.getWidth();
    const int height = color.getHeight();
    std::cout << "Resolution: " << width << "x" << height << std::endl;

    // Initialize the output image
    ImageBuffer output(width, height, 3);

    // Load the filter weights if specified
    std::vector<char> weights;
    if (!weightsFilename.empty())
    {
      std::cout << "Loading filter weights" << std::endl;
      weights = loadFile(weightsFilename);
    }

    // Initialize the denoising filter
    std::cout << "Initializing" << std::endl;
    Timer timer;

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

    oidn::FilterRef filter = device.newFilter(filterType.c_str());

    filter.setImage("color", color.getData(), oidn::Format::Float3, width, height);
    if (albedo)
      filter.setImage("albedo", albedo.getData(), oidn::Format::Float3, width, height);
    if (normal)
      filter.setImage("normal", normal.getData(), oidn::Format::Float3, width, height);

    filter.setImage("output", output.getData(), oidn::Format::Float3, width, height);

    if (hdr)
      filter.set("hdr", true);
    if (srgb)
      filter.set("srgb", true);

    if (maxMemoryMB >= 0)
      filter.set("maxMemoryMB", maxMemoryMB);

    if (!weights.empty())
      filter.setData("weights", weights.data(), weights.size());

    const bool showProgress = !ref && numBenchmarkRuns == 0 && verbose <= 2;
    if (showProgress)
    {
      filter.setProgressMonitorFunction(progressCallback);
      signal(SIGINT, signalHandler);
    }

    filter.commit();

    const double initTime = timer.query();

    const int versionMajor = device.get<int>("versionMajor");
    const int versionMinor = device.get<int>("versionMinor");
    const int versionPatch = device.get<int>("versionPatch");

    std::cout << "  version=" << versionMajor << "." << versionMinor << "." << versionPatch
              << ", filter=" << filterType
              << ", msec=" << (1000. * initTime) << std::endl;

    // Denoise the image
    if (!showProgress)
      std::cout << "Denoising" << std::endl;
    timer.reset();

    filter.execute();

    const double denoiseTime = timer.query();
    if (showProgress)
      std::cout << std::endl;
    if (verbose <= 2)
      std::cout << "  msec=" << (1000. * denoiseTime) << std::endl;

    if (showProgress)
    {
      filter.setProgressMonitorFunction(nullptr);
      signal(SIGINT, SIG_DFL);
    }

    if (!outputFilename.empty())
    {
      // Save output image
      std::cout << "Saving output" << std::endl;
      saveImage(outputFilename, output, srgb);
    }

    if (ref)
    {
      // Verify the output values
      std::cout << "Verifying output" << std::endl;

      int numErrors;
      float maxError;
      std::tie(numErrors, maxError) = compareImage(output, ref, 1e-4);

      std::cout << "  values=" << output.getSize() << ", errors=" << numErrors << ", maxerror=" << maxError << std::endl;

      if (numErrors > 0)
      {
        // Save debug images
        std::cout << "Saving debug images" << std::endl;
        saveImage("denoise_in.ppm",   color,  srgb);
        saveImage("denoise_out.ppm",  output, srgb);
        saveImage("denoise_ref.ppm",  ref,    srgb);

        throw std::runtime_error("output does not match the reference");
      }
    }

    if (numBenchmarkRuns > 0)
    {
      // Benchmark loop
    #ifdef VTUNE
      __itt_resume();
    #endif

      std::cout << "Benchmarking: " << "ntimes=" << numBenchmarkRuns << std::endl;
      timer.reset();

      for (int i = 0; i < numBenchmarkRuns; ++i)
        filter.execute();

      const double totalTime = timer.query();
      std::cout << "  sec=" << totalTime << ", msec/image=" << (1000.*totalTime / numBenchmarkRuns) << std::endl;

    #ifdef VTUNE
      __itt_pause();
    #endif
    }
  }
  catch (std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
