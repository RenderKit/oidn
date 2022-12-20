// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "tensor_layout.h"

namespace oidn {
  
  class Engine;
  class Buffer;
  class Filter;

  class Device : public RefCount, public Verbose
  {
  public:
    Device();

    static void setError(Device* device, Error code, const std::string& message);
    static Error getError(Device* device, const char** outMessage);
    void setErrorFunction(ErrorFunction func, void* userPtr);

    void warning(const std::string& message);

    virtual DeviceType getType() const = 0;

    virtual int get1i(const std::string& name);
    virtual void set1i(const std::string& name, int value);

    bool isCommitted() const { return committed; }
    void checkCommitted();
    void commit();

    OIDN_INLINE Device* getDevice() { return this; } // used by the API implementation
    OIDN_INLINE std::mutex& getMutex() { return mutex; }

    virtual Engine* getEngine(int i = 0) const = 0;
    virtual int getNumEngines() const = 0;

    // Native tensor layout
    DataType getTensorDataType() const { return tensorDataType; }
    TensorLayout getTensorLayout() const { return tensorLayout; }
    TensorLayout getWeightsLayout() const { return weightsLayout; }
    int getTensorBlockC() const { return tensorBlockSize; }

    // Memory
    virtual Storage getPointerStorage(const void* ptr) = 0;
    ExternalMemoryTypeFlags getExternalMemoryTypes() const { return externalMemoryTypes; }

    // Synchronizes all engines (does not block)
    virtual void submitBarrier() {}

    // Waits for all asynchronous commands to complete (blocks)
    virtual void wait() = 0;

    Ref<Filter> newFilter(const std::string& type);
   
  protected:
    virtual void init() = 0;

    // Native tensor layout
    DataType tensorDataType = DataType::Float32;
    TensorLayout tensorLayout = TensorLayout::chw;
    TensorLayout weightsLayout = TensorLayout::oihw;
    int tensorBlockSize = 1;

    ExternalMemoryTypeFlags externalMemoryTypes;

    // State
    bool dirty = true;
    bool committed = false;

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
  };

} // namespace oidn
