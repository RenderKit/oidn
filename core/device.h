// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace oidn {

  class Buffer;
  class Filter;

  class ScratchBuffer;
  class ScratchBufferManager;

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

    // Neural network runtime
  #if defined(OIDN_DNNL)
    dnnl::engine dnnlEngine;
    dnnl::stream dnnlStream;
  #endif
    int tensorBlockSize = 1;

    // Memory
    std::weak_ptr<ScratchBufferManager> scratchManagerWp;

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

    void warning(const std::string& message);

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

    Ref<ScratchBuffer> newScratchBuffer(size_t byteSize);

    __forceinline Device* getDevice() { return this; }
    __forceinline std::mutex& getMutex() { return mutex; }

  #if defined(OIDN_DNNL)
    __forceinline dnnl::engine& getDNNLEngine() { return dnnlEngine; }
    __forceinline dnnl::stream& getDNNLStream() { return dnnlStream; }
  #endif

    // Returns the native tensor layout block size
    __forceinline int getTensorBlockSize() const { return tensorBlockSize; }

    bool isCommitted() const { return bool(arena); }
    void checkCommitted();

  private:
    void print();
  };

} // namespace oidn
