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

#pragma once

#include <cstdio>
#include <algorithm>
#include <map>
#include "common/tensor.h"

namespace oidn {

  // Loads the contents of a file into a buffer
  static Ref<Buffer> load_file(const char* filename)
  {
    FILE* file = fopen(filename, "rb");
    if (!file)
      throw std::runtime_error("cannot open file");

    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0, SEEK_SET);

    Ref<Buffer> buffer = make_ref<Buffer>(size);
    if (fread(buffer->data(), 1, size, file) != size)
      throw std::runtime_error("read error");

    fclose(file);
    return buffer;
  }

  // Loads images stored in a tensor archive
  static std::map<std::string, Tensor> load_images_tza(const char* filename)
  {
    Ref<Buffer> buffer = load_file(filename);
    std::map<std::string, Tensor> images = parse_tensors(buffer->data());

    // Add refs to the buffer
    for (auto& i : images)
      i.second.buffer = buffer;

    return images;
  }

  // Saves a 3-channel image in HWC format to a PPM file
  static void save_image_ppm(const Tensor& image, const char* filename)
  {
    if (image.ndims() != 3 || image.dims[2] != 3 || image.format != "hwc")
      throw std::invalid_argument("image must have 3 channels");

    FILE* file = fopen(filename, "wb");
    if (!file)
      throw std::runtime_error("cannot create image");

    fprintf(file, "P6\n%d %d\n255\n", image.dims[1], image.dims[0]);

    for (int i = 0; i < image.size(); ++i)
    {
      int c = std::min(std::max(int(image[i] * 255.f), 0), 255);
      fputc(c, file);
    }

    fclose(file);
  }

} // ::oidn

