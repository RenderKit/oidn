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

#include <cmath>
#include <algorithm>
#include <fstream>
#include "image_io.h"

namespace oidn {

  Tensor loadImagePFM(const std::string& filename)
  {
    // Open the file
    std::ifstream file(filename, std::ios::binary);
    if (file.fail())
      throw std::runtime_error("cannot open file '" + filename + "'");

    // Read the header
    std::string id;
    file >> id;
    int C;
    if (id == "PF")
      C = 3;
    else if (id == "Pf")
      C = 1;
    else
      throw std::runtime_error("invalid PFM image");

    int H, W;
    file >> W >> H;

    float scale;
    file >> scale;

    file.get(); // skip newline

    if (file.fail())
      throw std::runtime_error("invalid PFM image");

    if (scale >= 0.f)
      throw std::runtime_error("big-endian PFM images are not supported");
    scale = fabs(scale);

    // Read the pixels
    Tensor image({H, W, C}, "hwc");

    for (int h = 0; h < H; ++h)
    {
      for (int w = 0; w < W; ++w)
      {
        for (int c = 0; c < C; ++c)
        {
          float x;
          file.read((char*)&x, sizeof(float));
          image[((H-1-h)*W + w) * C + c] = x * scale;
        }
      }
    }

    if (file.fail())
      throw std::runtime_error("invalid PFM image");

    return image;
  }

  void saveImagePFM(const Tensor& image, const std::string& filename)
  {
    if (image.ndims() != 3 || image.dims[2] != 3 || image.format != "hwc")
      throw std::invalid_argument("image must have 3 channels");
    const int H = image.dims[0];
    const int W = image.dims[1];
    const int C = image.dims[2];

    // Open the file
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
      throw std::runtime_error("cannot open file: '" + filename + "'");

    // Write the header
    file << "PF" << std::endl;
    file << W << " " << H << std::endl;
    file << "-1.0" << std::endl;

    // Write the pixels
    for (int h = 0; h < H; ++h)
    {
      for (int w = 0; w < W; ++w)
      {
        for (int c = 0; c < 3; ++c)
        {
          const float x = image[((H-1-h)*W + w) * C + c];
          file.write((char*)&x, sizeof(float));
        }
      }
    }
  }

  void saveImagePPM(const Tensor& image, const std::string& filename)
  {
    if (image.ndims() != 3 || image.dims[2] != 3 || image.format != "hwc")
      throw std::invalid_argument("image must have 3 channels");
    const int H = image.dims[0];
    const int W = image.dims[1];
    const int C = image.dims[2];

    // Open the file
    std::ofstream file(filename, std::ios::binary);
    if (file.fail())
      throw std::runtime_error("cannot open file: '" + filename + "'");

    // Write the header
    file << "P6" << std::endl;
    file << W << " " << H << std::endl;
    file << "255" << std::endl;

    // Write the pixels
    for (int i = 0; i < W*H; ++i)
    {
      for (int c = 0; c < 3; ++c)
      {
        float x = image[i*C+c];
        x = pow(x, 1.f/2.2f);
        int ch = std::min(std::max(int(x * 255.f), 0), 255);
        file.put(char(ch));
      }
    }
  }

} // namespace oidn

