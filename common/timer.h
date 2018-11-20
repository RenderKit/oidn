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
    double invCountsPerSec;
    LARGE_INTEGER startCount;
  #elif defined(__APPLE__)
    double invCountsPerSec;
    uint64_t startTime;
  #else
    timespec startTime;
  #endif

  public:
    Timer()
    {
  #if defined(_WIN32)
      LARGE_INTEGER frequency;

      BOOL result = QueryPerformanceFrequency(&frequency);
      assert(result != 0 && "timer is not supported");

      invCountsPerSec = 1.0 / (double)frequency.QuadPart;
  #elif defined(__APPLE__)
      mach_timebaseInfo_data_t timebaseInfo;
      mach_timebaseInfo(&timebaseInfo);
      invCountsPerSec = (double)timebaseInfo.numer / (double)timebaseInfo.denom * 1e-9;
  #endif

      reset();
    }

    void reset()
    {
  #if defined(_WIN32)
      BOOL result = QueryPerformanceCounter(&startCount);
      assert(result != 0 && "could not query counter");
  #elif defined(__APPLE__)
      startTime = mach_absolute_time();
  #else
      int result = clock_gettime(CLOCK_MONOTONIC, &startTime);
      MAYBE_UNUSED(result);
      assert(result == 0 && "could not get time");
  #endif
    }

    double query() const
    {
  #if defined(_WIN32)
      LARGE_INTEGER currentCount;

      BOOL result = QueryPerformanceCounter(&currentCount);
      assert(result != 0 && "could not query counter");

      return (currentCount.QuadPart - startCount.QuadPart) * invCountsPerSec;
  #elif defined(__APPLE__)
      uint64_t endTime = mach_absolute_time();
      uint64_t elapsedTime = endTime - startTime;
      return (double)elapsedTime * invCountsPerSec;
  #else
      timespec currentTime;

      int result = clock_gettime(CLOCK_MONOTONIC, &currentTime);
      MAYBE_UNUSED(result);
      assert(result == 0 && "could not get time");

      return (double)(currentTime.tv_sec - startTime.tv_sec) + (double)(currentTime.tv_nsec - startTime.tv_nsec) * 1e-9;
  #endif
    }
  };

} // ::oidn
