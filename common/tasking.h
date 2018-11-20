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

#define TBB_PREVIEW_LOCAL_OBSERVER 1
#include "tbb/task_scheduler_init.h"
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_arena.h"
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

namespace oidn {

  class ThreadAffinity
  {
  private:
    std::vector<cpu_set_t> cpusets;    // thread affinities
    std::vector<cpu_set_t> oldCpusets; // original thread affinities

  public:
    ThreadAffinity(int threadsPerCore = INT_MAX);

    int numThreads() const
    {
      if (cpusets.empty())
        return tbb::this_task_arena::max_concurrency();
      return (int)cpusets.size();
    }

    // Sets the affinity (0..num_threads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);
  };

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

} // ::oidn
