// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module.h"
#include "device_factory.h"
#include <mutex>
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
    }

    // Initializes the global context (should be called by API functions)
    void init()
    {
      std::call_once(initFlag, [this]()
      {
        getEnvVar("OIDN_VERBOSE", verbose);

        // Load the modules
      #if defined(OIDN_DEVICE_CPU)
        if (getEnvVarOrDefault("OIDN_DEVICE_CPU", 1))
          OIDN_INIT_STATIC_MODULE(device_cpu);
      #endif
      #if defined(OIDN_DEVICE_SYCL)
        if (getEnvVarOrDefault("OIDN_DEVICE_SYCL", 1))
          OIDN_INIT_MODULE(device_sycl);
      #endif
      #if defined(OIDN_DEVICE_CUDA)
        if (getEnvVarOrDefault("OIDN_DEVICE_CUDA", 1))
          OIDN_INIT_MODULE(device_cuda);
      #endif
      #if defined(OIDN_DEVICE_HIP)
        if (getEnvVarOrDefault("OIDN_DEVICE_HIP", 1))
          OIDN_INIT_MODULE(device_hip);
      #endif
      #if defined(OIDN_DEVICE_METAL)
        if (getEnvVarOrDefault("OIDN_DEVICE_METAL", 1))
          OIDN_INIT_STATIC_MODULE(device_metal);
      #endif
      #if defined(OIDN_METAL_IOS)
        getEnvVarOrDefault("OIDN_METAL_IOS", 1);
          // OIDN_INIT_STATIC_MODULE(device_metal);
      #endif

        // Sort the physical devices by score
        std::sort(physicalDevices.begin(), physicalDevices.end(),
                  [](const Ref<PhysicalDevice>& a, const Ref<PhysicalDevice>& b)
                  { return a->score > b->score; });
      });
    }

    bool isDeviceSupported(DeviceType type) const;
    DeviceFactory* getDeviceFactory(DeviceType type) const;
    int getNumPhysicalDevices() const { return static_cast<int>(physicalDevices.size()); }
    const Ref<PhysicalDevice>& getPhysicalDevice(int id) const;

    Ref<Device> newDevice(int physicalDeviceID);

  private:
    Context() = default;
    Context(const Context&) = delete;
    Context& operator =(const Context&) = delete;

    std::once_flag initFlag;
    ModuleLoader modules;
    std::map<DeviceType, std::unique_ptr<DeviceFactory>> deviceFactories;
    std::vector<Ref<PhysicalDevice>> physicalDevices;
  };

OIDN_NAMESPACE_END
