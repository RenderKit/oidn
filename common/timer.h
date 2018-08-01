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

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#elif defined(__APPLE__)
#include <mach/mach_time.h>
#else
#include <ctime>
#endif

namespace oidn {

  class Timer
  {
  private:
  #if defined(_WIN32)
    double inv_counts_per_sec;
    LARGE_INTEGER start_count;
  #elif defined(__APPLE__)
    double inv_counts_per_sec;
    uint64_t start_time;
  #else
    timespec start_time;
  #endif

  public:
    Timer()
    {
  #if defined(_WIN32)
      LARGE_INTEGER frequency;

      BOOL result = QueryPerformanceFrequency(&frequency);
      assert(result != 0 && "timer is not supported");

      inv_counts_per_sec = 1.0 / (double)frequency.QuadPart;
  #elif defined(__APPLE__)
      mach_timebase_info_data_t timebase_info;
      mach_timebase_info(&timebase_info);
      inv_counts_per_sec = (double)timebase_info.numer / (double)timebase_info.denom * 1e-9;
  #endif

      reset();
    }

    void reset()
    {
  #if defined(_WIN32)
      BOOL result = QueryPerformanceCounter(&start_count);
      assert(result != 0 && "could not query counter");
  #elif defined(__APPLE__)
      start_time = mach_absolute_time();
  #else
      int result = clock_gettime(CLOCK_MONOTONIC, &start_time);
      assert(result == 0 && "could not get time");
  #endif
    }

    double query() const
    {
  #if defined(_WIN32)
      LARGE_INTEGER current_count;

      BOOL result = QueryPerformanceCounter(&current_count);
      assert(result != 0 && "could not query counter");

      return (current_count.QuadPart - start_count.QuadPart) * inv_counts_per_sec;
  #elif defined(__APPLE__)
      uint64_t end_time = mach_absolute_time();
      uint64_t elapsed_time = end_time - start_time;
      return (double)elapsed_time * inv_counts_per_sec;
  #else
      timespec current_time;

      int result = clock_gettime(CLOCK_MONOTONIC, &current_time);
      assert(result == 0 && "could not get time");

      return (double)(current_time.tv_sec - start_time.tv_sec) + (double)(current_time.tv_nsec - start_time.tv_nsec) * 1e-9;
  #endif
    }
  };

} // ::oidn
