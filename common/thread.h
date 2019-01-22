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

#include "platform.h"
#if defined(__APPLE__)
  #include <mach/thread_policy.h>
#endif

#include <vector>

namespace oidn {

#if defined(_WIN32)

  // --------------------------------------------------------------------------
  // ThreadAffinity - Windows
  // --------------------------------------------------------------------------

  class ThreadAffinity
  {
  private:
    typedef BOOL (WINAPI *GetLogicalProcessorInformationExFunc)(LOGICAL_PROCESSOR_RELATIONSHIP,
                                                                PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                                                                PDWORD);

    typedef BOOL (WINAPI *SetThreadGroupAffinityFunc)(HANDLE,
                                                      CONST GROUP_AFFINITY*,
                                                      PGROUP_AFFINITY);

    GetLogicalProcessorInformationExFunc pGetLogicalProcessorInformationEx = nullptr;
    SetThreadGroupAffinityFunc pSetThreadGroupAffinity = nullptr;

    std::vector<GROUP_AFFINITY> affinities;    // thread affinities
    std::vector<GROUP_AFFINITY> oldAffinities; // original thread affinities

  public:
    ThreadAffinity(int numThreadsPerCore = INT_MAX);

    int getNumThreads() const
    {
      return (int)affinities.size();
    }

    // Sets the affinity (0..numThreads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);
  };

#elif defined(__linux__)

  // --------------------------------------------------------------------------
  // ThreadAffinity - Linux
  // --------------------------------------------------------------------------

  class ThreadAffinity
  {
  private:
    std::vector<cpu_set_t> affinities;    // thread affinities
    std::vector<cpu_set_t> oldAffinities; // original thread affinities

  public:
    ThreadAffinity(int numThreadsPerCore = INT_MAX);

    int getNumThreads() const
    {
      return (int)affinities.size();
    }

    // Sets the affinity (0..numThreads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);
  };

#elif defined(__APPLE__)

  // --------------------------------------------------------------------------
  // ThreadAffinity - macOS
  // --------------------------------------------------------------------------

  class ThreadAffinity
  {
  private:
    std::vector<thread_affinity_policy> affinities;    // thread affinities
    std::vector<thread_affinity_policy> oldAffinities; // original thread affinities

  public:
    ThreadAffinity(int numThreadsPerCore = INT_MAX);

    int getNumThreads() const
    {
      return (int)affinities.size();
    }

    // Sets the affinity (0..numThreads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);
  };

#endif

} // namespace oidn
