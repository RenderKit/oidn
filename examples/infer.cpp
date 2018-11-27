// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

using namespace std;
using namespace oidn;

int main(int argc, char **argv)
{
  try
  {
    // Load input image
    std::string inputFilename = "test0.tza";
    if (argc > 1)
      inputFilename = argv[1];
    std::string refFilename = inputFilename.substr(0, inputFilename.find_last_of('.')) + "_ref.tza";
    if (argc > 2)
      refFilename = argv[2];

    Tensor input = loadImageTZA(inputFilename);
    const int H = input.dims[0];
    const int W = input.dims[1];
    cout << inputFilename << ": " << W << "x" << H << endl;

    Tensor output({ H, W, 3 }, "hwc");

    // Initialize denoising filter
    Timer timer;

    oidn::DeviceRef device = oidn::newDevice();
    oidn::FilterRef filter = device.newFilter("Autoencoder");

    const size_t F = sizeof(float);
    filter.setImage("color",  input.data,  oidn::Format::Float3, W, H, 0 * F, 9 * F);
    filter.setImage("albedo", input.data,  oidn::Format::Float3, W, H, 3 * F, 9 * F);
    filter.setImage("normal", input.data,  oidn::Format::Float3, W, H, 6 * F, 9 * F);
    filter.setImage("output", output.data, oidn::Format::Float3, W, H, 0 * F, 3 * F);
    filter.set1i("srgb", 1);

    filter.commit();

    const char* errorMessage;
    if (device.getError(&errorMessage) != oidn::Error::None)
    {
      cout << "error: " << errorMessage << endl;
      return 1;
    }

    double initTime = timer.query();
    cout << "init=" << (1000. * initTime) << " msec" << endl;

    // Load reference output image
    Tensor ref = loadImageTZA(refFilename);
    if (ref.dims != output.dims)
    {
      cout << "error: reference output size mismatch" << endl;
      return 1;
    }

    // Correctness check and warmup
    filter.execute();
    int nerr = 0;
    float maxre = 0;
    for (size_t i = 0; i < output.size(); ++i)
    {
      float expect = std::max(ref.data[i], 0.f);
      float actual = std::max(output.data[i], 0.f);
      float re;
      if (abs(expect) < 1e-5 && abs(actual) < 1e-5)
        re = 0;
      else if (expect != 0)
        re = abs((expect - actual) / expect);
      else
        re = abs(expect - actual);
      if (maxre < re) maxre = re;
      if (re > 1e-5)
      {
        //cout << "i=" << i << " expect=" << expect << " actual=" << actual << endl;
        ++nerr;
      }
    }
    cout << "checked " << output.size() << " floats, nerr=" << nerr << ", maxre=" << maxre << endl;

    // Save images
    saveImagePPM(output, "infer_out.ppm");
    saveImagePPM(ref, "infer_ref.ppm");
    saveImagePPM(input, "infer_in.ppm");

    // Benchmark loop
#ifdef VTUNE
    __itt_resume();
#endif
    cout << "===== start =====" << endl;
    timer.reset();
    int ntimes = 100;
    //int ntimes = 5;
    for (int i = 0; i < ntimes; ++i)
      filter.execute();
    double totalTime = timer.query();
    cout << "===== stop =====" << endl;
    cout << "ntimes=" << ntimes << " secs=" << totalTime
      << " msec/image=" << (1000.*totalTime / ntimes) << endl;
#ifdef VTUNE
    __itt_pause();
#endif
  }
  catch (std::exception& e)
  {
    cout << "error: " << e.what() << endl;
    return 1;
  }

  return 0;
}
