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

#include "common/mkldnn.h"
#include "common/ref.h"
#include "common/buffer.h"
#include "common/tasking.h"
#include "include/OpenImageDenoise/oidn.hpp"

namespace oidn {

  using OIDN::Format;
  using OIDN::BufferType;
  using OIDN::FilterType;


  inline memory::dims tensor_dims(const std::shared_ptr<memory>& mem)
  {
    const mkldnn_memory_desc_t& md = mem->get_primitive_desc().desc().data;
    return memory::dims(&md.dims[0], &md.dims[md.ndims]);
  }

  inline memory::data_type tensor_type(const std::shared_ptr<memory>& mem)
  {
    const mkldnn_memory_desc_t& md = mem->get_primitive_desc().desc().data;
    return memory::data_type(md.data_type);
  }

  // Returns the number of values in a tensor
  inline size_t tensor_size(const memory::dims& dims)
  {
    size_t res = 1;
    for (int i = 0; i < dims.size(); ++i)
      res *= dims[i];
    return res;
  }

  inline size_t tensor_size(const std::shared_ptr<memory>& mem)
  {
    return tensor_size(tensor_dims(mem));
  }


  template<int K>
  inline int padded(int dim)
  {
    return (dim + (K-1)) & ~(K-1);
  }

  template<int K>
  inline memory::dims padded_dims_nchw(const memory::dims& dims)
  {
    assert(dims.size() == 4);
    memory::dims pad_dims = dims;
    pad_dims[1] = padded<K>(dims[1]); // pad C
    return pad_dims;
  }


  template<int K>
  struct BlockedFormat;

  template<>
  struct BlockedFormat<8>
  {
    static constexpr memory::format nChwKc   = memory::format::nChw8c;
    static constexpr memory::format OIhwKiKo = memory::format::OIhw8i8o;
  };

  template<>
  struct BlockedFormat<16>
  {
    static constexpr memory::format nChwKc   = memory::format::nChw16c;
    static constexpr memory::format OIhwKiKo = memory::format::OIhw16i16o;
  };


  __forceinline float linear_to_srgb(float x)
  {
    return std::pow(x, 1.f/2.2f);
  }

  __forceinline float srgb_to_linear(float x)
  {
    return std::pow(x, 2.2f);
  }

} // ::oidn
