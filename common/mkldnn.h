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

#include "platform.h"

#include "mkl-dnn/include/mkldnn.hpp"
#include "mkl-dnn/include/mkldnn_debug.h"
#include "mkl-dnn/src/common/utils.hpp"
#include "mkl-dnn/src/cpu/jit_generator.hpp"

namespace oidn {

  using namespace mkldnn;
  using namespace mkldnn::impl::utils;
  using namespace mkldnn::impl::cpu;

} // ::oidn
