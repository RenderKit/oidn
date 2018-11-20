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

#include "common.h"

namespace oidn {

class LinearTransferFunction
{
public:
  __forceinline float forward(float x) const { return x; }
  __forceinline float reverse(float x) const { return x; }
};

class SrgbTransferFunction
{
public:
  __forceinline float forward(float x) const { return linear_to_srgb(x); }
  __forceinline float reverse(float x) const { return srgb_to_linear(x); }
};

} // ::oidn
