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

#include "tasking.h"
#include <fstream>

namespace oidn {

  ThreadAffinity::ThreadAffinity(int threads_per_core)
  {
    std::vector<int> thread_ids;

    // Parse the thread/CPU topology
    for (int cpu_id = 0; ; cpu_id++)
    {
      std::fstream fs;
      std::string cpu = std::string("/sys/devices/system/cpu/cpu") + std::to_string(cpu_id) + std::string("/topology/thread_siblings_list");
      fs.open(cpu.c_str(), std::fstream::in);
      if (fs.fail()) break;

      int i;
      int j = 0;
      while ((j < threads_per_core) && (fs >> i))
      {
        if (std::none_of(thread_ids.begin(), thread_ids.end(), [&](int id) { return id == i; }))
          thread_ids.push_back(i);

        if (fs.peek() == ',')
          fs.ignore();
        j++;
      }

      fs.close();
    }

  #if 0
    for (size_t i = 0; i < thread_ids.size(); ++i)
      std::cout << "thread " << i << " -> " << thread_ids[i] << std::endl;
  #endif

    // Create the cpusets
    cpusets.resize(thread_ids.size());
    for (size_t i = 0; i < thread_ids.size(); ++i)
    {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(thread_ids[i], &cpuset);
      cpusets[i] = cpuset;
    }

    old_cpusets.resize(thread_ids.size());
    for (size_t i = 0; i < old_cpusets.size(); ++i)
      CPU_ZERO(&old_cpusets[i]);
  }

  void ThreadAffinity::set(int thread_index)
  {
    if (thread_index >= (int)cpusets.size())
      return;

    const pthread_t thread = pthread_self();

    // Save the current affinity
    if (pthread_getaffinity_np(thread, sizeof(cpu_set_t), &old_cpusets[thread_index]) != 0)
    {
      WARNING("pthread_getaffinity_np failed");
      old_cpusets[thread_index] = cpusets[thread_index];
      return;
    }

    // Set the new affinity
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpusets[thread_index]) != 0)
      WARNING("pthread_setaffinity_np failed");
  }

  void ThreadAffinity::restore(int thread_index)
  {
    if (thread_index >= (int)cpusets.size())
      return;

    const pthread_t thread = pthread_self();

    // Restore the original affinity
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &old_cpusets[thread_index]) != 0)
      WARNING("pthread_setaffinity_np failed");
  }


  PinningObserver::PinningObserver(const std::shared_ptr<ThreadAffinity>& affinity)
    : affinity(affinity)
  {
    observe(true);
  }

  PinningObserver::PinningObserver(const std::shared_ptr<ThreadAffinity>& affinity, tbb::task_arena& arena)
    : tbb::task_scheduler_observer(arena),
      affinity(affinity)
  {
    observe(true);
  }

  PinningObserver::~PinningObserver()
  {
    observe(false);
  }

  void PinningObserver::on_scheduler_entry(bool is_worker)
  {
    const int thread_index = tbb::this_task_arena::current_thread_index();
    affinity->set(thread_index);
  }

  void PinningObserver::on_scheduler_exit(bool is_worker)
  {
    const int thread_index = tbb::this_task_arena::current_thread_index();
    affinity->restore(thread_index);
  }

} // ::oidn
