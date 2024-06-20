// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module.h"
#include "device_factory.h"
#include <mutex>
#include <set>
#include <map>

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_STATIC_LIB)
  OIDN_DECLARE_INIT_STATIC_MODULE(device_cpu);
  OIDN_DECLARE_INIT_STATIC_MODULE(device_metal);

  #define OIDN_INIT_STATIC_MODULE(name) init_##name()
#else
  #define OIDN_INIT_STATIC_MODULE(name) modules.load(#name)
#endif

#define OIDN_INIT_MODULE(name) modules.load(#name)

  // Global library context
  class Context : public Verbose
  {
  public:
    // Returns the global context without initialization
    static Context& get();

    // Registers a device type (should be called by device modules)
    template<typename DeviceFactoryT>
    static void registerDeviceType(DeviceType type, const std::vector<Ref<PhysicalDevice>>& physicalDevices)
    {
      if (physicalDevices.empty())
        return;

      Context& ctx = get();
      ctx.deviceFactories[type] = std::unique_ptr<DeviceFactory>(new DeviceFactoryT);

      // Add the detected physical devices to the context
      for (const auto& physicalDevice : physicalDevices)
        ctx.physicalDevices.push_back(physicalDevice);

      // Sort all physical devices detected so far by score
      std::sort(ctx.physicalDevices.begin(), ctx.physicalDevices.end(),
                [](const Ref<PhysicalDevice>& a, const Ref<PhysicalDevice>& b)
                { return a->score > b->score; });
    }

    Context()
    {
      getEnvVar("OIDN_VERBOSE", verbose);
    }

    // The context is *not* thread-safe, so a mutex is provided for synchronization
    std::mutex& getMutex() { return mutex; }

    // Initializes the context (called only by API functions)
    // If called with a non-default device type, initializes only that device type
    void init(DeviceType deviceType = DeviceType::Default)
    {
      if (fullyInited || initedDeviceTypes.find(deviceType) != initedDeviceTypes.end())
        return;

      // Initialize the device modules, if necessary
    #if defined(OIDN_DEVICE_CPU)
      if (initDeviceType(deviceType, DeviceType::CPU, "OIDN_DEVICE_CPU"))
        OIDN_INIT_STATIC_MODULE(device_cpu);
    #endif
    #if defined(OIDN_DEVICE_SYCL)
      if (initDeviceType(deviceType, DeviceType::SYCL, "OIDN_DEVICE_SYCL"))
        OIDN_INIT_MODULE(device_sycl);
    #endif
    #if defined(OIDN_DEVICE_CUDA)
      if (initDeviceType(deviceType, DeviceType::CUDA, "OIDN_DEVICE_CUDA"))
        OIDN_INIT_MODULE(device_cuda);
    #endif
    #if defined(OIDN_DEVICE_HIP)
      if (initDeviceType(deviceType, DeviceType::HIP, "OIDN_DEVICE_HIP"))
        OIDN_INIT_MODULE(device_hip);
    #endif
    #if defined(OIDN_DEVICE_METAL)
      if (initDeviceType(deviceType, DeviceType::Metal, "OIDN_DEVICE_METAL"))
        OIDN_INIT_STATIC_MODULE(device_metal);
    #endif

      if (deviceType == DeviceType::Default)
        fullyInited = true;
    }

    bool isDeviceSupported(DeviceType type) const;
    DeviceFactory* getDeviceFactory(DeviceType type) const;
    int getNumPhysicalDevices() const { return static_cast<int>(physicalDevices.size()); }
    const Ref<PhysicalDevice>& getPhysicalDevice(int id) const;

    Ref<Device> newDevice(int physicalDeviceID);
    Ref<Device> newDevice(DeviceType type);

  private:
    Context(const Context&) = delete;
    Context& operator =(const Context&) = delete;

    bool initDeviceType(DeviceType deviceType, DeviceType targetDeviceType,
                        const std::string& envVar)
    {
      assert(targetDeviceType != DeviceType::Default);
      if ((deviceType != targetDeviceType && deviceType != DeviceType::Default) ||
          initedDeviceTypes.find(targetDeviceType) != initedDeviceTypes.end())
        return false;

      initedDeviceTypes.insert(targetDeviceType); // try to initialize only once
      return getEnvVarOrDefault(envVar, 1);
    }

    std::mutex mutex;
    bool fullyInited = false;
    std::set<DeviceType> initedDeviceTypes;
    ModuleLoader modules;
    std::map<DeviceType, std::unique_ptr<DeviceFactory>> deviceFactories;
    std::vector<Ref<PhysicalDevice>> physicalDevices;
  };

OIDN_NAMESPACE_END