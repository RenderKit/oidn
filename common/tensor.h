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

#include "buffer.h"
#include <vector>
#include <map>

namespace oidn {

  // Generic tensor
  struct Tensor
  {
    float* data;
    std::vector<int> dims;
    std::string format;
    Ref<Buffer> buffer; // optional, only for reference

    __forceinline Tensor() : data(nullptr) {}
    __forceinline int ndims() const { return (int)dims.size(); }

    // Returns the number of values
    __forceinline size_t size() const
    {
      size_t size = 1;
      for (int i = 0; i < ndims(); ++i)
        size *= dims[i];
      return size;
    }

    __forceinline float& operator [](size_t i) { return data[i]; }
    __forceinline const float& operator [](size_t i) const { return data[i]; }
  };

  // Parses tensors from a buffer
  std::map<std::string, Tensor> parse_tensors(void* buffer);

} // ::oidn
