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

#if defined(_MSC_VER)
  #pragma warning (disable : 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

#include "tasking.h"
#include <fstream>

namespace oidn {

#if defined(_WIN32)

  // Windows
  ThreadAffinity::ThreadAffinity(int threadsPerCore)
  {
    HMODULE hLib = GetModuleHandle(TEXT("kernel32"));
    pGetLogicalProcessorInformationEx = (GetLogicalProcessorInformationExFunc)GetProcAddress(hLib, "GetLogicalProcessorInformationEx");
    pSetThreadGroupAffinity = (SetThreadGroupAffinityFunc)GetProcAddress(hLib, "SetThreadGroupAffinity");

    if (pGetLogicalProcessorInformationEx && pSetThreadGroupAffinity)
    {
      // Get logical processor information
      PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = nullptr;
      DWORD bufferSize = 0;

      // First call the function with an empty buffer to get the required buffer size
      BOOL result = pGetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &bufferSize);
      if (result || GetLastError() != ERROR_INSUFFICIENT_BUFFER)
      {
        WARNING("GetLogicalProcessorInformationEx failed");
        return;
      }
      
      // Allocate the buffer
      buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(bufferSize);
      if (!buffer)
      {
        WARNING("SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX allocation failed");
        return;
      }
        
      // Call again the function but now with the properly sized buffer
      result = pGetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &bufferSize);
      if (!result)
      {
        WARNING("GetLogicalProcessorInformationEx failed");
        free(buffer);
        return;
      }

      // Iterate over the logical processor information structures
      // There should be one structure for each physical core
      char* ptr = (char*)buffer;
      while (ptr < (char*)buffer + bufferSize)
      {
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX item = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)ptr;
        if (item->Relationship == RelationProcessorCore && item->Processor.GroupCount > 0)
        {
          // Iterate over the groups
          int numThreads = 0;
          for (int group = 0; (group < item->Processor.GroupCount) && (numThreads < threadsPerCore); ++group)
          {
            GROUP_AFFINITY coreAffinity = item->Processor.GroupMask[group];
            while ((coreAffinity.Mask != 0) && (numThreads < threadsPerCore))
            {
              // Extract the next set bit/thread from the mask
              GROUP_AFFINITY threadAffinity = coreAffinity;
              threadAffinity.Mask = threadAffinity.Mask & -threadAffinity.Mask;

              // Push the affinity for this thread
              affinities.push_back(threadAffinity);
              oldAffinities.push_back(threadAffinity);
              numThreads++;

              // Remove this bit/thread from the mask
              coreAffinity.Mask ^= threadAffinity.Mask;
            }
          }
        }

        // Next structure
        ptr += item->Size;
      }

      // Free the buffer
      free(buffer);
    }
  }

  void ThreadAffinity::set(int threadIndex)
  {
    if (threadIndex >= (int)affinities.size())
      return;

    // Save the current affinity and set the new one
    const HANDLE thread = GetCurrentThread();
    if (!pSetThreadGroupAffinity(thread, &affinities[threadIndex], &oldAffinities[threadIndex]))
      WARNING("SetThreadGroupAffinity failed");
  }

  void ThreadAffinity::restore(int threadIndex)
  {
    if (threadIndex >= (int)affinities.size())
      return;

    // Restore the original affinity
    const HANDLE thread = GetCurrentThread();
    if (!pSetThreadGroupAffinity(thread, &oldAffinities[threadIndex], nullptr))
      WARNING("SetThreadGroupAffinity failed");
  }

#else

  // Linux
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

    // Create the affinity structures
    affinities.resize(threadIds.size());
    oldAffinities.resize(threadIds.size());

    for (size_t i = 0; i < threadIds.size(); ++i)
    {
      cpu_set_t affinity;
      CPU_ZERO(&affinity);
      CPU_SET(threadIds[i], &affinity);

      affinities[i] = affinity;
      oldAffinities[i] = affinity;
    }
  }

  void ThreadAffinity::set(int threadIndex)
  {
    if (threadIndex >= (int)affinities.size())
      return;

    const pthread_t thread = pthread_self();

    // Save the current affinity
    if (pthread_getaffinity_np(thread, sizeof(cpu_set_t), &oldAffinities[threadIndex]) != 0)
    {
      WARNING("pthread_getaffinity_np failed");
      oldAffinities[threadIndex] = affinities[threadIndex];
      return;
    }

    // Set the new affinity
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &affinities[threadIndex]) != 0)
      WARNING("pthread_setaffinity_np failed");
  }

  void ThreadAffinity::restore(int threadIndex)
  {
    if (threadIndex >= (int)affinities.size())
      return;

    const pthread_t thread = pthread_self();

    // Restore the original affinity
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &oldAffinities[threadIndex]) != 0)
      WARNING("pthread_setaffinity_np failed");
  }

#endif


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

} // namespace oidn
