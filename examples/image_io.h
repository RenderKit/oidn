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

#pragma once

#include <string>
#include "common/tensor.h"

namespace oidn {

  // Loads an image from a PFM file
  Tensor loadImagePFM(const std::string& filename);

  // Saves an image to a PFM file
  void saveImagePFM(const Tensor& image, const std::string& filename);

  // Saves an image to a PPM file
  void saveImagePPM(const Tensor& image, const std::string& filename);

} // namespace oidn

