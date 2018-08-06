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
#include <sys/time.h>
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
  OIDN::Device device = OIDN::newDevice();

  std::string input_filename = "test0.tza";
  std::string refoutput_filename = "test0_refout.tza";
  if (argc > 2)
  {
      input_filename = argv[1];
      refoutput_filename = argv[2];
  }

  Tensor input = load_image_tza(input_filename);
  const int H = input.dims[0];
  const int W = input.dims[1];
  cout << input_filename << ": " << W << "x" << H << endl;

  Tensor output({H, W, 3}, "hwc");

  Timer timer;

  OIDN::Filter filter = device.newFilter(OIDN::FilterType::AUTOENCODER_LDR);

  const size_t F = sizeof(float);
  filter.setBuffer(OIDN::BufferType::INPUT,        0, OIDN::Format::FLOAT3_SRGB, input.data,  0*F, 9*F, W, H);
  filter.setBuffer(OIDN::BufferType::INPUT_ALBEDO, 0, OIDN::Format::FLOAT3,      input.data,  3*F, 9*F, W, H);
  filter.setBuffer(OIDN::BufferType::INPUT_NORMAL, 0, OIDN::Format::FLOAT3,      input.data,  6*F, 9*F, W, H);
  filter.setBuffer(OIDN::BufferType::OUTPUT,       0, OIDN::Format::FLOAT3_SRGB, output.data, 0*F, 3*F, W, H);

  filter.commit();

  double initd = timer.query();
  cout << "init=" << (1000. * initd) << " msec" << endl;

  // correctness check and warmup
  Tensor refoutput = load_image_tza(refoutput_filename);
  filter.execute();
  int nerr = 0;
  float maxre = 0;
  for (size_t i = 0; i < output.size(); ++i)
  {
    float expect = refoutput.data[i];
    float actual = output.data[i];
    float re;
    if (abs(expect) < 1e-5 && abs(actual) < 1e-5)
      re = 0;
    else if (expect != 0)
      re = abs((expect-actual)/expect);
    else
      re = abs(expect-actual);
    if (maxre < re) maxre = re;
    if (re > 1e-5)
    {
      //cout << "i=" << i << " expect=" << expect << " actual=" << actual << endl;
      ++nerr;
    }
  }
  cout << "checked " << output.size() << " floats, nerr=" << nerr << ", maxre=" << maxre << endl;

  // save images
  save_image_ppm(output,    "infer_out.ppm");
  save_image_ppm(refoutput, "infer_refout.ppm");
  save_image_ppm(input,     "infer_in.ppm");

  // benchmark loop
  #ifdef VTUNE
  __itt_resume();
  #endif
  double mind = INFINITY;
  struct timeval tv1, tv2;
  gettimeofday(&tv1, 0);
  cout << "===== start =====" << endl;
  int ntimes = 100;
  for (int i = 0; i < ntimes; ++i)
  {
    timer.reset();
    filter.execute();
    mind = min(mind, timer.query());
  }
  gettimeofday(&tv2, 0);
  cout << "===== stop =====" << endl;
  double d1 = tv1.tv_sec + (double)tv1.tv_usec * 1e-6;
  double d2 = tv2.tv_sec + (double)tv2.tv_usec * 1e-6;
  cout << "ntimes=" << ntimes << " secs=" << (d2-d1)
       << " msec/image=" << (1000.*(d2-d1)/ntimes)
       << " (min=" << (1000.*mind) << ")" << endl;
  #ifdef VTUNE
  __itt_pause();
  #endif

  return 0;
}
