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
#include <vector>

// We need to define these to avoid implicit linkage against
// tbb_debug.lib under Windows. When removing these lines debug build
// under Windows fails.
#define __TBB_NO_IMPLICIT_LINKAGE 1
#define __TBBMALLOC_NO_IMPLICIT_LINKAGE 1

#define TBB_PREVIEW_LOCAL_OBSERVER 1
#include "tbb/task_scheduler_init.h"
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_arena.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

namespace oidn {

#if defined(_WIN32)

  // Windows
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
    ThreadAffinity(int threadsPerCore = INT_MAX);

    int numThreads() const
    {
      if (affinities.empty())
        return tbb::this_task_arena::max_concurrency();
      return (int)affinities.size();
    }

    // Sets the affinity (0..num_threads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);
  };

#else

  // Linux
  class ThreadAffinity
  {
  private:
    std::vector<cpu_set_t> affinities;    // thread affinities
    std::vector<cpu_set_t> oldAffinities; // original thread affinities

  public:
    ThreadAffinity(int threadsPerCore = INT_MAX);

    int numThreads() const
    {
      if (affinities.empty())
        return tbb::this_task_arena::max_concurrency();
      return (int)affinities.size();
    }

    // Sets the affinity (0..num_threads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);
  };

#endif

  class PinningObserver : public tbb::task_scheduler_observer
  {
  private:
    std::shared_ptr<ThreadAffinity> affinity;

  public:
    explicit PinningObserver(const std::shared_ptr<ThreadAffinity>& affinity);
    PinningObserver(const std::shared_ptr<ThreadAffinity>& affinity, tbb::task_arena& arena);
    ~PinningObserver();

    void on_scheduler_entry(bool isWorker) override;
    void on_scheduler_exit(bool isWorker) override;
  };

} // namespace oidn
