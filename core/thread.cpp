// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#if defined(_MSC_VER)
  #pragma warning (disable : 4146) // unary minus operator applied to unsigned type, result still unsigned
#endif

#if defined(__linux__)
  #include <sched.h>
  #include <unordered_set>
#elif defined(__APPLE__)
  #include <mach/thread_act.h>
  #include <mach/mach_init.h>
#endif

#include "thread.h"
#include <fstream>

OIDN_NAMESPACE_BEGIN

#if defined(_WIN32)

  // -----------------------------------------------------------------------------------------------
  // ThreadAffinity: Windows
  // -----------------------------------------------------------------------------------------------

  ThreadAffinity::ThreadAffinity(int maxNumThreadsPerCore, int verbose)
    : Verbose(verbose)
  {
    HMODULE hLib = GetModuleHandle(TEXT("kernel32"));
    pGetLogicalProcessorInformationEx =
      (GetLogicalProcessorInformationExFunc)GetProcAddress(hLib, "GetLogicalProcessorInformationEx");
    pSetThreadGroupAffinity =
      (SetThreadGroupAffinityFunc)GetProcAddress(hLib, "SetThreadGroupAffinity");
    if (!pGetLogicalProcessorInformationEx || !pSetThreadGroupAffinity)
      return;

    // Get logical processor information
    PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX buffer = nullptr;
    DWORD bufferSize = 0;

    // First call the function with an empty buffer to get the required buffer size
    BOOL result = pGetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &bufferSize);
    if (result || GetLastError() != ERROR_INSUFFICIENT_BUFFER)
    {
      printWarning("GetLogicalProcessorInformationEx failed");
      return;
    }

    // Allocate the buffer
    buffer = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)malloc(bufferSize);
    if (!buffer)
    {
      printWarning("SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX allocation failed");
      return;
    }

    // Call again the function but now with the properly sized buffer
    result = pGetLogicalProcessorInformationEx(RelationProcessorCore, buffer, &bufferSize);
    if (!result)
    {
      printWarning("GetLogicalProcessorInformationEx failed");
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
        int numThreadsPerCore = 0;
        for (int group = 0; group < item->Processor.GroupCount &&
                            numThreadsPerCore < maxNumThreadsPerCore; ++group)
        {
          GROUP_AFFINITY coreAffinity = item->Processor.GroupMask[group];
          while (coreAffinity.Mask != 0 && numThreadsPerCore < maxNumThreadsPerCore)
          {
            // Extract the next set bit/thread from the mask
            GROUP_AFFINITY threadAffinity = coreAffinity;
            threadAffinity.Mask = threadAffinity.Mask & -threadAffinity.Mask;

            // Push the affinity for this thread
            affinities.push_back(threadAffinity);
            oldAffinities.push_back(threadAffinity);
            numThreadsPerCore++;

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

  void ThreadAffinity::set(int threadIndex)
  {
    if (threadIndex >= (int)affinities.size())
      return;

    // Save the current affinity and set the new one
    const HANDLE thread = GetCurrentThread();
    if (!pSetThreadGroupAffinity(thread, &affinities[threadIndex], &oldAffinities[threadIndex]))
      printWarning("SetThreadGroupAffinity failed");
  }

  void ThreadAffinity::restore(int threadIndex)
  {
    if (threadIndex >= (int)affinities.size())
      return;

    // Restore the original affinity
    const HANDLE thread = GetCurrentThread();
    if (!pSetThreadGroupAffinity(thread, &oldAffinities[threadIndex], nullptr))
      printWarning("SetThreadGroupAffinity failed");
  }

#elif defined(__linux__)

  // -----------------------------------------------------------------------------------------------
  // ThreadAffinity: Linux
  // -----------------------------------------------------------------------------------------------

  ThreadAffinity::ThreadAffinity(int maxNumThreadsPerCore, int verbose)
    : Verbose(verbose)
  {
    // Get the process affinity mask
    cpu_set_t processAffinity;
    if (sched_getaffinity(0, sizeof(cpu_set_t), &processAffinity) != 0)
    {
      printWarning("sched_getaffinity failed");
      return;
    }

    // Parse the thread/CPU topology
    std::vector<int> threadIDs;
    std::unordered_set<int> visitedThreadIDs;

    for (int cpuID = 0; ; cpuID++)
    {
      const std::vector<int> siblingIDs = parseList(
        "/sys/devices/system/cpu/cpu" + std::to_string(cpuID) + "/topology/thread_siblings_list");
      if (siblingIDs.empty())
        break;

      int numThreadsPerCore = 0;
      for (int siblingID : siblingIDs)
      {
        if (visitedThreadIDs.find(siblingID) == visitedThreadIDs.end())
        {
          visitedThreadIDs.insert(siblingID);
          if (numThreadsPerCore < maxNumThreadsPerCore && CPU_ISSET(siblingID, &processAffinity))
          {
            threadIDs.push_back(siblingID);
            numThreadsPerCore++;
          }
        }
      }
    }

  #if 0
    for (size_t i = 0; i < thread_ids.size(); ++i)
      std::cout << "thread " << i << " -> " << thread_ids[i] << std::endl;
  #endif

    // Create the affinity structures
    affinities.resize(threadIDs.size());
    oldAffinities.resize(threadIDs.size());

    for (size_t i = 0; i < threadIDs.size(); ++i)
    {
      cpu_set_t affinity;
      CPU_ZERO(&affinity);
      CPU_SET(threadIDs[i], &affinity);

      affinities[i] = affinity;
      oldAffinities[i] = affinity;
    }
  }

  void ThreadAffinity::set(int threadIndex)
  {
    if (threadIndex < 0 || threadIndex >= (int)affinities.size())
      return;

    const pthread_t thread = pthread_self();

    // Save the current affinity
    if (pthread_getaffinity_np(thread, sizeof(cpu_set_t), &oldAffinities[threadIndex]) != 0)
    {
      printWarning("pthread_getaffinity_np failed");
      oldAffinities[threadIndex] = affinities[threadIndex];
      return;
    }

    // Set the new affinity
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &affinities[threadIndex]) != 0)
      printWarning("pthread_setaffinity_np failed");
  }

  void ThreadAffinity::restore(int threadIndex)
  {
    if (threadIndex < 0 || threadIndex >= (int)affinities.size())
      return;

    const pthread_t thread = pthread_self();

    // Restore the original affinity
    if (pthread_setaffinity_np(thread, sizeof(cpu_set_t), &oldAffinities[threadIndex]) != 0)
      printWarning("pthread_setaffinity_np failed");
  }

  std::vector<int> ThreadAffinity::parseList(const std::string& filename)
  {
    std::vector<int> list;
    std::fstream fs(filename.c_str(), std::fstream::in);
    if (fs.fail())
      return list;

    int id = -1;
    while (fs >> id)
    {
      const int nextChar = fs.peek();
      if (nextChar == '-')
      {
        fs.ignore();
        int idEnd;
        if (!(fs >> idEnd))
          break;
        for (int i = id; i <= idEnd; ++i)
          list.push_back(i);
      }
      else
      {
        if (nextChar == ',')
          fs.ignore();
        list.push_back(id);
      }
    }

    return list;
  }

#elif defined(__APPLE__)

  // -----------------------------------------------------------------------------------------------
  // ThreadAffinity: macOS
  // -----------------------------------------------------------------------------------------------

  ThreadAffinity::ThreadAffinity(int maxNumThreadsPerCore, int verbose)
    : Verbose(verbose)
  {
    // Query the thread/CPU topology
    int numPhysicalCPUs;
    int numLogicalCPUs;

    if (!getSysctl("hw.physicalcpu", numPhysicalCPUs) || !getSysctl("hw.logicalcpu", numLogicalCPUs))
    {
      printWarning("sysctlbyname failed");
      return;
    }

    if (numLogicalCPUs % numPhysicalCPUs != 0 && maxNumThreadsPerCore > 1)
      return; // hybrid, not supported
    const int numThreadsPerCore = min(numLogicalCPUs / numPhysicalCPUs, maxNumThreadsPerCore);

    // Create the affinity structures
    // macOS doesn't support binding a thread to a specific core, but we can at least group threads which
    // should be on the same core together
    for (int core = 1; core <= numPhysicalCPUs; ++core) // tags start from 1!
    {
      thread_affinity_policy affinity;
      affinity.affinity_tag = core;

      for (int thread = 0; thread < numThreadsPerCore; ++thread)
      {
        affinities.push_back(affinity);
        oldAffinities.push_back(affinity);
      }
    }
  }

  void ThreadAffinity::set(int threadIndex)
  {
    if (threadIndex >= (int)affinities.size())
      return;

    const auto thread = mach_thread_self();

    // Save the current affinity
    mach_msg_type_number_t policyCount = THREAD_AFFINITY_POLICY_COUNT;
    boolean_t getDefault = FALSE;
    if (thread_policy_get(thread, THREAD_AFFINITY_POLICY,
                          (thread_policy_t)&oldAffinities[threadIndex],
                          &policyCount, &getDefault) != KERN_SUCCESS)
    {
      printWarning("thread_policy_get failed");
      oldAffinities[threadIndex] = affinities[threadIndex];
      return;
    }

    // Set the new affinity
    if (thread_policy_set(thread, THREAD_AFFINITY_POLICY,
                          (thread_policy_t)&affinities[threadIndex],
                          THREAD_AFFINITY_POLICY_COUNT) != KERN_SUCCESS)
      printWarning("thread_policy_set failed");
  }

  void ThreadAffinity::restore(int threadIndex)
  {
    if (threadIndex >= (int)affinities.size())
      return;

    const auto thread = mach_thread_self();

    // Restore the original affinity
    if (thread_policy_set(thread, THREAD_AFFINITY_POLICY,
                          (thread_policy_t)&oldAffinities[threadIndex],
                          THREAD_AFFINITY_POLICY_COUNT) != KERN_SUCCESS)
      printWarning("thread_policy_set failed");
  }

#endif

OIDN_NAMESPACE_END
