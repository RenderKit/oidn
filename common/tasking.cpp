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

  ThreadAffinity::ThreadAffinity(int threadsPerCore)
  {
    std::vector<int> threadIds;

    // Parse the thread/CPU topology
    for (int cpuId = 0; ; cpuId++)
    {
      std::fstream fs;
      std::string cpu = std::string("/sys/devices/system/cpu/cpu") + std::to_string(cpuId) + std::string("/topology/thread_siblings_list");
      fs.open(cpu.c_str(), std::fstream::in);
      if (fs.fail()) break;

      int i;
      int j = 0;
      while ((j < threadsPerCore) && (fs >> i))
      {
        if (std::none_of(threadIds.begin(), threadIds.end(), [&](int id) { return id == i; }))
          threadIds.push_back(i);

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
    cpusets.resize(threadIds.size());
    for (size_t i = 0; i < threadIds.size(); ++i)
    {
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(threadIds[i], &cpuset);
      cpusets[i] = cpuset;
    }

    oldCpusets.resize(threadIds.size());
    for (size_t i = 0; i < oldCpusets.size(); ++i)
      CPU_ZERO(&oldCpusets[i]);
  }

  void ThreadAffinity::set(int threadIndex)
  {
    if (threadIndex >= (int)cpusets.size())
      return;

    const pthread_t thread = pthread_self();

    // Save the current affinity
    if (pthread_getaffinity_np(thread, sizeof(cpu_set_t), &oldCpusets[threadIndex]) != 0)
    {
      WARNING("pthread_getaffinity_np failed");
      oldCpusets[threadIndex] = cpusets[threadIndex];
      return;
    }

    // Set the new affinity
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpusets[threadIndex]) != 0)
      WARNING("pthread_setaffinity_np failed");
  }

  void ThreadAffinity::restore(int threadIndex)
  {
    if (threadIndex >= (int)cpusets.size())
      return;

    const pthread_t thread = pthread_self();

    // Restore the original affinity
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &oldCpusets[threadIndex]) != 0)
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

  void PinningObserver::on_scheduler_entry(bool isWorker)
  {
    const int threadIndex = tbb::this_task_arena::current_thread_index();
    affinity->set(threadIndex);
  }

  void PinningObserver::on_scheduler_exit(bool isWorker)
  {
    const int threadIndex = tbb::this_task_arena::current_thread_index();
    affinity->restore(threadIndex);
  }

} // ::oidn
