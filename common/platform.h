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

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#endif

#include <xmmintrin.h>
#include <atomic>
#include <algorithm>
#include <memory>
#include <cmath>
#include <string>
#include <iostream>
#include <cassert>
#include "include/OpenImageDenoise/oidn.hpp"

#if defined(_WIN32)
  // Windows
#else
  // Linux
  #if !defined(__forceinline)
    #define __forceinline  inline __attribute__((always_inline))
  #endif
#endif

#ifndef UNUSED
  #define UNUSED(x) ((void)x)
#endif
#ifndef MAYBE_UNUSED
  #define MAYBE_UNUSED(x) UNUSED(x)
#endif

#define WARNING(x) { std::cerr << "Warning: " << x << std::endl << std::flush; }

