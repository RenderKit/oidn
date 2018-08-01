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

  const char* images_filename = "image0.tza";
  if (argc > 1)
      images_filename = argv[1];

  std::map<std::string, Tensor> images = load_images_tza(images_filename);
  auto input = images["input"];
  const int H = input.dims[0];
  const int W = input.dims[1];
  printf("%s: %dx%d\n", images_filename, W, H);

  size_t input_size = H*W*9;
  size_t output_size = H*W*3;
  printf("input_size %lu output_size %lu\n", input_size, output_size);
  assert(input.size() == input_size);
  std::vector<float> output(output_size);

  Timer timer;

  OIDN::Filter filter = device.newFilter(OIDN::FilterType::AUTOENCODER_LDR);

  const size_t F = sizeof(float);
  filter.setBuffer(OIDN::BufferType::INPUT,        0, OIDN::Format::FLOAT3_SRGB, input.data,    0*F, 9*F, W, H);
  filter.setBuffer(OIDN::BufferType::INPUT_ALBEDO, 0, OIDN::Format::FLOAT3,      input.data,    3*F, 9*F, W, H);
  filter.setBuffer(OIDN::BufferType::INPUT_NORMAL, 0, OIDN::Format::FLOAT3,      input.data,    6*F, 9*F, W, H);
  filter.setBuffer(OIDN::BufferType::OUTPUT,       0, OIDN::Format::FLOAT3_SRGB, output.data(), 0*F, 3*F, W, H);

  filter.commit();

  double initd = timer.query();
  printf("init: %g ms\n", 1000. * initd);

  // correctness check and warmup
  auto refoutput = images["output"];
  filter.execute();
  int nerr = 0;
  float maxre = 0;
  for (size_t i = 0; i < output_size; ++i)
  {
    float expect = refoutput.data[i];
    float actual = output[i];
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
      //printf("i=%d expect=%g actual=%g\n", i, expect, actual);
      ++nerr;
    }
  }
  printf("Checked %lu floats, nerr = %d, maxre = %g\n",
      output_size, nerr, maxre);

  // save output image
  Tensor output_tensor;
  output_tensor.dims = {input.dims[0], input.dims[1], 3};
  output_tensor.format = "hwc";
  output_tensor.data = output.data();
  save_image_ppm(output_tensor, "infer.ppm");

  // benchmark loop
  #ifdef VTUNE
  __itt_resume();
  #endif
  double mind = INFINITY;
  struct timeval tv1, tv2;
  gettimeofday(&tv1, 0);
  printf("===== start =====\n");
  int ntimes = 100;
  for (int i = 0; i < ntimes; ++i)
  {
    timer.reset();
    filter.execute();
    mind = min(mind, timer.query());
  }
  gettimeofday(&tv2, 0);
  printf("===== stop =====\n");
  double d1 = tv1.tv_sec + (double)tv1.tv_usec * 1e-6;
  double d2 = tv2.tv_sec + (double)tv2.tv_usec * 1e-6;
  printf("ntimes=%d secs=%g msec/image=%g (min=%g)\n", ntimes, d2-d1,
      1000.*(d2-d1)/ntimes, 1000.*mind);
  #ifdef VTUNE
  __itt_pause();
  #endif

  return 0;
}
