// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include "ref.h"
#include "exception.h"
#include "verbose.h"
#include "thread.h"
#include "tensor_layout.h"
#include "data.h"

OIDN_NAMESPACE_BEGIN

  class Engine;
  class Buffer;
  class Filter;

  class PhysicalDevice : public RefCount
  {
  public:
    DeviceType type = DeviceType::Default;
    int score = -1; // higher score *probably* means faster device

    std::string name = "Unknown";

    bool uuidSupported = false;
    UUID uuid{};

    bool luidSupported = false;
    LUID luid{};
    uint32_t nodeMask = 0;

    bool pciAddressSupported = false;
    int pciDomain   = 0;
    int pciBus      = 0;
    int pciDevice   = 0;
    int pciFunction = 0;

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

    // Some devices (e.g. CUDA, HIP) need to change some per-thread state, which must be later restored
    // Most device calls must be between enter() and leave() calls
    virtual void enter() {}
    virtual void leave() {}

    virtual DeviceType getType() const = 0;

    virtual int getInt(const std::string& name);
    virtual void setInt(const std::string& name, int value);

    bool isCommitted() const { return committed; }
    void checkCommitted();
    void commit();

    Ref<Filter> newFilter(const std::string& type);

    oidn_inline Device* getDevice() { return this; } // used by the API implementation
    oidn_inline std::mutex& getMutex() { return mutex; }

    virtual Engine* getEngine(int i = 0) const = 0;
    virtual int getNumEngines() const = 0;

    // Native tensor layout
    DataType getTensorDataType() const { return tensorDataType; }
    DataType getWeightDataType() const { return weightDataType; }
    TensorLayout getTensorLayout() const { return tensorLayout; }
    TensorLayout getWeightLayout() const { return weightLayout; }
    int getTensorBlockC() const { return tensorBlockC; }
    int getMinTileAlignment() const { return minTileAlignment; }
    virtual bool needWeightAndBiasOnDevice() const { return true; }

    // Memory
    virtual Storage getPtrStorage(const void* ptr) { return Storage::Undefined; }
    bool isSystemMemorySupported()  const { return systemMemorySupported; }
    bool isManagedMemorySupported() const { return managedMemorySupported; }
    ExternalMemoryTypeFlags getExternalMemoryTypes() const { return externalMemoryTypes; }
    void trimScratch();

    // Synchronizes all engines (does not block)
    virtual void submitBarrier() {}

    // Issues all previously submitted commands (does not block)
    virtual void flush() {}

    // Waits for all previously submitted commands to complete (blocks)
    virtual void wait() = 0;

  protected:
    virtual void init() = 0;

    // Native tensor layout
    DataType tensorDataType = DataType::Float32;
    DataType weightDataType = DataType::Float32;
    TensorLayout tensorLayout = TensorLayout::chw;
    TensorLayout weightLayout = TensorLayout::oihw;
    int tensorBlockC = 1;
    int minTileAlignment = 1; // minimum spatial tile alignment in pixels

    bool systemMemorySupported  = false;
    bool managedMemorySupported = false;
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
