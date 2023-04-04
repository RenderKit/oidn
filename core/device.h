// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include "ref.h"
#include "exception.h"
#include "thread.h"
#include "tensor_layout.h"
#include "data.h"

OIDN_NAMESPACE_BEGIN
  
  class Engine;
  class Buffer;
  class Filter;

  // Synchronization mode for operations
  enum class SyncMode
  {
    Sync, // synchronous
    Async // asynchronous
  };

  class PhysicalDevice : public RefCount
  {
  public:
    DeviceType type = DeviceType::Default;
    int score = -1; // higher score *probably* means faster device

    std::string name = "Unknown";

    UUID uuid{};
    bool uuidValid = false;
    
    LUID luid{};
    bool luidValid = false;
    uint32_t nodeMask = 0;

    PhysicalDevice(DeviceType type, int score) : type(type), score(score) {}

    virtual int getInt(const std::string& name) const;
    virtual const char* getString(const std::string& name) const;
    virtual Data getData(const std::string& name) const;
  };

  class Device : public RefCount, public Verbose
  {
  public:
    Device();

    static void setError(Device* device, Error code, const std::string& message);
    static Error getError(Device* device, const char** outMessage);
    void setErrorFunction(ErrorFunction func, void* userPtr);

    void warning(const std::string& message);

    // Some devices (e.g. CUDA, HIP) need to change some per-thread state, which must be later restored
    // Most device calls must be between begin() and end() calls
    virtual void begin() {}
    virtual void end() {}

    virtual DeviceType getType() const = 0;

    virtual int getInt(const std::string& name);
    virtual void setInt(const std::string& name, int value);

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
    TensorLayout getWeightLayout() const { return weightLayout; }
    int getTensorBlockC() const { return tensorBlockC; }

    // Memory
    virtual Storage getPointerStorage(const void* ptr) = 0;
    ExternalMemoryTypeFlags getExternalMemoryTypes() const { return externalMemoryTypes; }
    virtual bool isMemoryUsageLimitSupported() const { return true; }

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
    TensorLayout weightLayout = TensorLayout::oihw;
    int tensorBlockC = 1;

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

  // SYCL devices require additional methods exposed for the API implementation
  class SYCLDeviceBase : public Device
  {
  public:
    virtual void setDepEvents(const sycl::event* events, int numEvents) = 0;
    virtual void getDoneEvent(sycl::event& event) = 0;
  };

OIDN_NAMESPACE_END
