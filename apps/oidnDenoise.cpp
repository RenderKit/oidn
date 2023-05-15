// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common/common.h"
#include "common/timer.h"
#include "utils/arg_parser.h"
#include "utils/image_io.h"
#include "utils/device_info.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cassert>
#include <limits>
#include <cmath>
#include <signal.h>
#ifdef VTUNE
#include <ittnotify.h>
#endif

OIDN_NAMESPACE_USING

void printUsage()
{
  std::cout << "Intel(R) Open Image Denoise" << std::endl;
  std::cout << "usage: oidnDenoise [-d/--device [0-9]+|default|cpu|sycl|cuda|hip]" << std::endl
            << "                   [-f/--filter RT|RTLightmap]" << std::endl
            << "                   [--hdr color.pfm] [--ldr color.pfm] [--srgb] [--dir directional.pfm]" << std::endl
            << "                   [--alb albedo.pfm] [--nrm normal.pfm] [--clean_aux]" << std::endl
            << "                   [--is/--input_scale value]" << std::endl
            << "                   [-o/--output output.pfm] [-r/--ref reference_output.pfm]" << std::endl
            << "                   [-t/--type float|half]" << std::endl
            << "                   [-q/--quality default|h|high|b|balanced]" << std::endl
            << "                   [-w/--weights weights.tza]" << std::endl
            << "                   [--threads n] [--affinity 0|1] [--maxmem MB] [--inplace]" << std::endl
            << "                   [-n times_to_run] [-v/--verbose 0-3]" << std::endl
            << "                   [--ld|--list_devices] [-h/--help]" << std::endl;
}

void errorCallback(void* userPtr, Error error, const char* message)
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
  {
    std::cout << std::endl;
    return false;
  }
  std::cout << "\rDenoising " << int(n * 100.) << "%" << std::flush;
  return true;
}

std::vector<char> loadFile(const std::string& filename)
{
  std::ifstream file(filename, std::ios::binary);
  if (file.fail())
    throw std::runtime_error("cannot open file: '" + filename + "'");
  file.seekg(0, file.end);
  const size_t size = file.tellg();
  file.seekg(0, file.beg);
  std::vector<char> buffer(size);
  file.read(buffer.data(), size);
  if (file.fail())
    throw std::runtime_error("error reading from file: '" + filename + "'");
  return buffer;
}

int main(int argc, char* argv[])
{
  DeviceType deviceType = DeviceType::Default;
  PhysicalDeviceRef physicalDevice;
  std::string filterType = "RT";
  std::string colorFilename, albedoFilename, normalFilename;
  std::string outputFilename, refFilename;
  std::string weightsFilename;
  Quality quality = Quality::Default;
  bool hdr = false;
  bool srgb = false;
  bool directional = false;
  float inputScale = std::numeric_limits<float>::quiet_NaN();
  bool cleanAux = false;
  Format dataType = Format::Undefined;
  int numRuns = 1;
  int numThreads = -1;
  int setAffinity = -1;
  int maxMemoryMB = -1;
  bool inplace = false;
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
      if (opt == "d" || opt == "dev" || opt == "device")
      {
        std::string value = args.getNext();
        if (isdigit(value[0]))
          physicalDevice = fromString<int>(value);
        else
          deviceType = fromString<DeviceType>(value);
      }
      else if (opt == "f" || opt == "filter")
        filterType = args.getNextValue();
      else if (opt == "hdr")
      {
        colorFilename = args.getNextValue();
        hdr = true;
      }
      else if (opt == "ldr")
      {
        colorFilename = args.getNextValue();
        hdr = false;
      }
      else if (opt == "srgb")
        srgb = true;
      else if (opt == "dir")
      {
        colorFilename = args.getNextValue();
        directional = true;
      }
      else if (opt == "alb" || opt == "albedo")
        albedoFilename = args.getNextValue();
      else if (opt == "nrm" || opt == "normal")
        normalFilename = args.getNextValue();
      else if (opt == "o" || opt == "out" || opt == "output")
        outputFilename = args.getNextValue();
      else if (opt == "r" || opt == "ref" || opt == "reference")
        refFilename = args.getNextValue();
      else if (opt == "is" || opt == "input_scale" || opt == "input-scale" || opt == "inputScale" || opt == "inputscale")
        inputScale = args.getNextValue<float>();
      else if (opt == "clean_aux" || opt == "clean-aux" || opt == "cleanAux" || opt == "cleanaux")
        cleanAux = true;
      else if (opt == "t" || opt == "type")
      {
        const auto val = toLower(args.getNextValue());
        if (val == "f" || val == "float" || val == "fp32")
          dataType = Format::Float;
        else if (val == "h" || val == "half" || val == "fp16")
          dataType = Format::Half;
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
        else
          throw std::runtime_error("invalid filter quality mode");
      }
      else if (opt == "w" || opt == "weights")
        weightsFilename = args.getNextValue();
      else if (opt == "n")
        numRuns = std::max(args.getNextValue<int>(), 1);
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
      else if (opt == "ld" || opt == "list_devices" || opt == "list-devices" || opt == "listDevices" || opt == "listdevices")
        return printPhysicalDevices();
      else if (opt == "h" || opt == "help")
      {
        printUsage();
        return 1;
      }
      else
        throw std::invalid_argument("invalid argument '" + opt + "'");
    }

  #if defined(OIDN_ARCH_X64)
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
  #endif

    // Initialize the denoising device
    std::cout << "Initializing device" << std::endl;
    Timer timer;

    DeviceRef device;
    if (physicalDevice)
      device = physicalDevice.newDevice();
    else
      device = newDevice(deviceType);

    const char* errorMessage;
    if (device.getError(errorMessage) != Error::None)
      throw std::runtime_error(errorMessage);
    device.setErrorFunction(errorCallback);

    if (numThreads > 0)
      device.set("numThreads", numThreads);
    if (setAffinity >= 0)
      device.set("setAffinity", bool(setAffinity));
    if (verbose >= 0)
      device.set("verbose", verbose);
    device.commit();

    const double deviceInitTime = timer.query();

    deviceType = device.get<DeviceType>("type");
    const int versionMajor = device.get<int>("versionMajor");
    const int versionMinor = device.get<int>("versionMinor");
    const int versionPatch = device.get<int>("versionPatch");

    std::cout << "  device=" << deviceType
              << ", version=" << versionMajor << "." << versionMinor << "." << versionPatch
              << ", msec=" << (1000. * deviceInitTime) << std::endl;

    // Load the input image
    std::shared_ptr<ImageBuffer> input, ref;
    std::shared_ptr<ImageBuffer> color, albedo, normal;

    std::cout << "Loading input" << std::endl;

    if (!albedoFilename.empty())
      input = albedo = loadImage(device, albedoFilename, 3, false, dataType);

    if (!normalFilename.empty())
      input = normal = loadImage(device, normalFilename, 3, dataType);

    if (!colorFilename.empty())
      input = color = loadImage(device, colorFilename, 3, srgb, dataType);

    if (!input)
      throw std::runtime_error("no input image specified");

    if (!refFilename.empty())
    {
      ref = loadImage(device, refFilename, 3, srgb, dataType);
      if (ref->getDims() != input->getDims())
        throw std::runtime_error("invalid reference output image");
    }

    const int width  = input->getW();
    const int height = input->getH();
    std::cout << "Resolution: " << width << "x" << height << std::endl;

    // Initialize the output image
    std::shared_ptr<ImageBuffer> output;
    if (inplace)
      output = input;
    else
      output = std::make_shared<ImageBuffer>(device, width, height, 3, input->getDataType());

    std::shared_ptr<ImageBuffer> inputCopy;
    if (inplace && numRuns > 1)
      inputCopy = input->clone();

    // Load the filter weights if specified
    std::vector<char> weights;
    if (!weightsFilename.empty())
    {
      std::cout << "Loading filter weights" << std::endl;
      weights = loadFile(weightsFilename);
    }

    // Initialize the denoising filter
    std::cout << "Initializing filter" << std::endl;
    timer.reset();

    FilterRef filter = device.newFilter(filterType.c_str());

    if (color)
      filter.setImage("color", color->getBuffer(), color->getFormat(), color->getW(), color->getH());
    if (albedo)
      filter.setImage("albedo", albedo->getBuffer(), albedo->getFormat(), albedo->getW(), albedo->getH());
    if (normal)
      filter.setImage("normal", normal->getBuffer(), normal->getFormat(), normal->getW(), normal->getH());

    filter.setImage("output", output->getBuffer(), output->getFormat(), output->getW(), output->getH());

    if (filterType == "RT")
    {
      if (hdr)
        filter.set("hdr", true);
      if (srgb)
        filter.set("srgb", true);
    }
    else if (filterType == "RTLightmap")
    {
      if (directional)
        filter.set("directional", true);
    }

    if (std::isfinite(inputScale))
      filter.set("inputScale", inputScale);

    if (cleanAux)
      filter.set("cleanAux", cleanAux);

    if (quality != Quality::Default)
      filter.set("quality", quality);

    if (maxMemoryMB >= 0)
      filter.set("maxMemoryMB", maxMemoryMB);

    if (!weights.empty())
      filter.setData("weights", weights.data(), weights.size());

    const bool showProgress = verbose <= 1;
    if (showProgress)
    {
      filter.setProgressMonitorFunction(progressCallback);
      signal(SIGINT, signalHandler);
    }

    filter.commit();

    const double filterInitTime = timer.query();

    std::cout << "  filter=" << filterType
              << ", msec=" << (1000. * filterInitTime) << std::endl;

    // Denoise the image
    uint32_t prevHash = 0;
    for (int run = 0; run < numRuns; ++run)
    {
      if (inplace && run > 0)
        memcpy(input->getData(), inputCopy->getData(), inputCopy->getByteSize());

      if (!showProgress)
        std::cout << "Denoising" << std::endl;
      timer.reset();

      filter.execute();

      const double denoiseTime = timer.query();

      if (showProgress)
        std::cout << std::endl;
      std::cout << "  msec=" << (1000. * denoiseTime);

      if (numRuns > 1 || verbose >= 2)
      {
        // Compute a hash of the output
        const size_t numBytes = output->getByteSize();
        const uint8_t* outputBytes = static_cast<const uint8_t*>(output->getData());
        uint32_t hash = 0x811c9dc5;
        for (size_t i = 0; i < numBytes; ++i)
        {
          hash ^= outputBytes[i];
          hash *= 0x1000193;
        }
        std::cout << ", hash=" << std::hex << std::setfill('0') << std::setw(8) << hash << std::dec << std::endl;

        if (run > 0 && hash != prevHash)
          throw std::runtime_error("output hash mismatch (non-deterministic output)");
        prevHash = hash;
      }
      else
        std::cout << std::endl;

      if (run == 0 && ref)
      {
        // Verify the output values
        std::cout << "Verifying output" << std::endl;

        const double errorThreshold = (input == normal || directional) ? 0.05 : 0.003;
        size_t numErrors;
        double avgError;
        std::tie(numErrors, avgError) = compareImage(*output, *ref, errorThreshold);

        std::cout << "  values=" << output->getSize()
                  << ", errors=" << numErrors << ", avgerror=" << avgError << std::endl;

        if (numErrors > 0)
        {
          // Save debug images
          std::cout << "Saving debug images" << std::endl;
          saveImage("denoise_in.pfm",  *input,  srgb);
          saveImage("denoise_out.pfm", *output, srgb);
          saveImage("denoise_ref.pfm", *ref,    srgb);

          throw std::runtime_error("output does not match the reference");
        }
      }
    }

    if (showProgress)
    {
      filter.setProgressMonitorFunction(nullptr);
      signal(SIGINT, SIG_DFL);
    }

    if (!outputFilename.empty())
    {
      // Save output image
      std::cout << "Saving output" << std::endl;
      saveImage(outputFilename, *output, srgb);
    }
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
