// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#include "verbose.h"

#if !defined(_WIN32)
  #include <pthread.h>
  #include <sched.h>
  #if defined(__APPLE__)
    #include <mach/thread_policy.h>
  #endif
#endif

#include <vector>
#include <mutex>

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // ThreadLocal
  // -----------------------------------------------------------------------------------------------

  // Wrapper which makes any variable thread-local
  template<typename T>
  class ThreadLocal : public Verbose
  {
  public:
    ThreadLocal(int verbose = 0)
      : Verbose(verbose)
    {
    #if defined(_WIN32)
      key = TlsAlloc();
      if (key == TLS_OUT_OF_INDEXES)
        throw std::runtime_error("TlsAlloc failed");
    #else
      if (pthread_key_create(&key, nullptr) != 0)
        throw std::runtime_error("pthread_key_create failed");
    #endif
    }

    ~ThreadLocal()
    {
      std::lock_guard<std::mutex> lock(mutex);
      for (T* ptr : instances)
        delete ptr;

    #if defined(_WIN32)
      if (!TlsFree(key))
        printWarning("TlsFree failed");
    #else
      if (pthread_key_delete(key) != 0)
        printWarning("pthread_key_delete failed");
    #endif
    }

    T& get()
    {
    #if defined(_WIN32)
      T* ptr = (T*)TlsGetValue(key);
    #else
      T* ptr = (T*)pthread_getspecific(key);
    #endif

      if (ptr)
        return *ptr;

      ptr = new T;
      std::lock_guard<std::mutex> lock(mutex);
      instances.push_back(ptr);

    #if defined(_WIN32)
      if (!TlsSetValue(key, ptr))
        throw std::runtime_error("TlsSetValue failed");
    #else
      if (pthread_setspecific(key, ptr) != 0)
        throw std::runtime_error("pthread_setspecific failed");
    #endif

      return *ptr;
    }

  private:
    // Disable copying
    ThreadLocal(const ThreadLocal&) = delete;
    ThreadLocal& operator =(const ThreadLocal&) = delete;

  #if defined(_WIN32)
    DWORD key;
  #else
    pthread_key_t key;
  #endif

    std::vector<T*> instances;
    std::mutex mutex;
  };

#if defined(_WIN32)

  // -----------------------------------------------------------------------------------------------
  // ThreadAffinity: Windows
  // -----------------------------------------------------------------------------------------------

  class ThreadAffinity : public Verbose
  {
  public:
    ThreadAffinity(int maxNumThreadsPerCore = INT_MAX, int verbose = 0);

    int getNumThreads() const
    {
      return (int)affinities.size();
    }

    // Sets the affinity (0..numThreads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);

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
  };

#elif defined(__linux__)

  // -----------------------------------------------------------------------------------------------
  // ThreadAffinity: Linux
  // -----------------------------------------------------------------------------------------------

  class ThreadAffinity : public Verbose
  {
  public:
    ThreadAffinity(int maxNumThreadsPerCore = INT_MAX, int verbose = 0);

    int getNumThreads() const
    {
      return (int)affinities.size();
    }

    // Sets the affinity (0..numThreads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);

  private:
    // Parses a list of numbers from a file in /sys/devices/system
    static std::vector<int> parseList(const std::string& filename);

    std::vector<cpu_set_t> affinities;    // thread affinities
    std::vector<cpu_set_t> oldAffinities; // original thread affinities
  };

#elif defined(__APPLE__)

  // -----------------------------------------------------------------------------------------------
  // ThreadAffinity: macOS
  // -----------------------------------------------------------------------------------------------

  class ThreadAffinity : public Verbose
  {
  public:
    ThreadAffinity(int maxNumThreadsPerCore = INT_MAX, int verbose = 0);

    int getNumThreads() const
    {
      return (int)affinities.size();
    }

    // Sets the affinity (0..numThreads-1) of the thread after saving the current affinity
    void set(int threadIndex);

    // Restores the affinity of the thread
    void restore(int threadIndex);

  private:
    std::vector<thread_affinity_policy> affinities;    // thread affinities
    std::vector<thread_affinity_policy> oldAffinities; // original thread affinities
  };

#endif

OIDN_NAMESPACE_END
