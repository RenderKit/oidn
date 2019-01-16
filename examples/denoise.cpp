// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include <iostream>
#include <cassert>
#include <cmath>

#ifdef VTUNE
#include <ittnotify.h>
#endif

#include "OpenImageDenoise/oidn.hpp"
#include "common/timer.h"
#include "image_io.h"
#include "cli.h"

using namespace std;
using namespace oidn;

void printUsage()
{
  cout << "Open Image Denoise Example" << endl;
  cout << "Usage: denoise [-ldr ldr_color.pfm] [-hdr hdr_color.pfm]" << endl
       << "               [-alb albedo.pfm] [-nrm normal.pfm]" << endl
       << "               [-o output.pfm] [-ref reference_output.pfm]" << endl
       << "               [-bench ntimes]" << endl;
}

int main(int argc, char* argv[])
{
  std::string colorFilename, albedoFilename, normalFilename;
  std::string outputFilename, refFilename;
  bool hdr = false;
  int benchmarkN = 0;

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
      if (opt == "ldr")
      {
        colorFilename = args.getNextValue();
        hdr = false;
      }
      else if (opt == "hdr")
      {
        colorFilename = args.getNextValue();
        hdr = true;
      }
      else if (opt == "alb" || opt == "albedo")
        albedoFilename = args.getNextValue();
      else if (opt == "nrm" || opt == "normal")
        normalFilename = args.getNextValue();
      else if (opt == "o" || opt == "out" || opt == "output")
        outputFilename = args.getNextValue();
      else if (opt == "ref" || opt == "reference")
        refFilename = args.getNextValue();
      else if (opt == "bench" || opt == "benchmark")
        benchmarkN = std::max(args.getNextValueInt(), 0);
      else if (opt == "h" || opt == "help")
      {
        printUsage();
        return 1;
      }
      else
        throw std::invalid_argument("invalid argument");
    }
  }
  catch (std::exception& e)
  {
    cout << "Error: " << e.what() << endl;
    return 1;
  }

  if (colorFilename.empty())
  {
    cout << "Error: no color image specified";
    return 1;
  }

  // Load the input image
  Tensor color, albedo, normal;
  Tensor ref;

  cout << "Loading input" << endl;

  color = loadImagePFM(colorFilename);
  if (!albedoFilename.empty())
    albedo = loadImagePFM(albedoFilename);
  if (!normalFilename.empty())
    normal = loadImagePFM(normalFilename);
  if (!refFilename.empty())
    ref = loadImagePFM(refFilename);

  const int H = color.dims[0];
  const int W = color.dims[1];
  cout << "Resolution: " << W << "x" << H << endl;

  // Initialize the output image
  Tensor output({H, W, 3}, "hwc");

  // Initialize the denoising filter
  cout << "Initializing";
  cout.flush();
  Timer timer;

  oidn::DeviceRef device = oidn::newDevice();
  oidn::FilterRef filter = device.newFilter("RT");

  filter.setImage("color", color.data, oidn::Format::Float3, W, H);
  if (albedo)
    filter.setImage("albedo", albedo.data, oidn::Format::Float3, W, H);
  if (normal)
    filter.setImage("normal", normal.data, oidn::Format::Float3, W, H);
  filter.setImage("output", output.data, oidn::Format::Float3, W, H);

  if (hdr)
    filter.set("hdr", true);

  filter.commit();

  const double initTime = timer.query();

  const int versionMajor = device.get<int>("versionMajor");
  const int versionMinor = device.get<int>("versionMinor");
  const int versionPatch = device.get<int>("versionPatch");

  const char* errorMessage;
  if (device.getError(&errorMessage) != oidn::Error::None)
  {
    cout << endl << "Error: " << errorMessage << endl;
    return 1;
  }

  cout << ": version=" << versionMajor << "." << versionMinor << "." << versionPatch
       << ", msec=" << (1000. * initTime) << endl;

  // Denoise the image
  cout << "Denoising";
  cout.flush();
  timer.reset();

  filter.execute();

  const double denoiseTime = timer.query();
  cout << ": msec=" << (1000. * denoiseTime) << endl;

  if (device.getError(&errorMessage) != oidn::Error::None)
  {
    cout << endl << "Error: " << errorMessage << endl;
    return 1;
  }

  if (ref)
  {
    if (ref.dims != output.dims)
    {
      cout << "Error: the reference image size does not match the input size" << endl;
      return 1;
    }

    // Verify the output values
    int nerr = 0;
    float maxre = 0;
    for (size_t i = 0; i < output.size(); ++i)
    {
      const float expect = std::max(ref.data[i], 0.f);
      const float actual = std::max(output.data[i], 0.f);
      float re;
      if (abs(expect) < 1e-5 && abs(actual) < 1e-5)
        re = 0;
      else if (expect != 0)
        re = abs((expect - actual) / expect);
      else
        re = abs(expect - actual);
      if (maxre < re) maxre = re;
      if (re > 1e-4)
      {
        //cout << "i=" << i << " expect=" << expect << " actual=" << actual << endl;
        ++nerr;
      }
    }
    cout << "Verified output: nfloats=" << output.size() << ", nerr=" << nerr << ", maxre=" << maxre << endl;

    // Save debug images
    cout << "Saving debug images" << endl;
    saveImagePPM(color,  "denoise.in.ppm");
    saveImagePPM(output, "denoise.out.ppm");
    saveImagePPM(ref,    "denoise.ref.ppm");
  }

  if (!outputFilename.empty())
  {
    // Save output image
    cout << "Saving output" << endl;
    saveImagePFM(output, outputFilename);
  }

  if (benchmarkN > 0)
  {
    // Benchmark loop
  #ifdef VTUNE
    __itt_resume();
  #endif

    cout << "Benchmarking: " << "ntimes=" << benchmarkN;
    cout.flush();
    timer.reset();

    for (int i = 0; i < benchmarkN; ++i)
      filter.execute();

    const double totalTime = timer.query();
    cout << ", sec=" << totalTime << ", msec/image=" << (1000.*totalTime / benchmarkN) << endl;

  #ifdef VTUNE
    __itt_pause();
  #endif
  }

  return 0;
}
