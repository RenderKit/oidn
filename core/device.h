// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace oidn {

  class Buffer;
  class Filter;

  class Device : public RefCount, public Verbose
  {
  private:
    // Thread-safety
    std::mutex mutex;

    // Error handling
    struct ErrorState
    {
      Error code = Error::None;
      std::string message;
    };

    static thread_local ErrorState globalError;
    ThreadLocal<ErrorState> error;
    ErrorFunction errorFunc = nullptr;
    void* errorUserPtr = nullptr;

    // Tasking
    std::shared_ptr<tbb::task_arena> arena;
    std::shared_ptr<PinningObserver> observer;
    std::shared_ptr<ThreadAffinity> affinity;

    // Parameters
    int numThreads = 0; // autodetect by default
    bool setAffinity = true;

    bool dirty = true;

  public:
    Device();
    ~Device();

    static void setError(Device* device, Error code, const std::string& message);
    static Error getError(Device* device, const char** outMessage);

    void setErrorFunction(ErrorFunction func, void* userPtr);

    int get1i(const std::string& name);
    void set1i(const std::string& name, int value);

    void commit();

    template<typename F>
    void executeTask(F& f)
    {
      arena->execute(f);
    }

    template<typename F>
    void executeTask(const F& f)
    {
      arena->execute(f);
    }

    Ref<Buffer> newBuffer(size_t byteSize);
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize);
    Ref<Filter> newFilter(const std::string& type);

    __forceinline Device* getDevice() { return this; }
    __forceinline std::mutex& getMutex() { return mutex; }

  private:
    bool isCommitted() const { return bool(arena); }
    void checkCommitted();

    void print();
  };

} // namespace oidn
