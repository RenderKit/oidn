// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module.h"
#include "device_factory.h"
#include <mutex>
#include <map>

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_STATIC_LIB)
#if defined(OIDN_DEVICE_CPU)
  void init_device_cpu();
#endif
#endif

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
      {
        // Prevent the physical device from being automatically destroyed to avoid issues at process
        // exit. This is needed because the physical device is owned by the context which is static,
        // thus it might get destroyed *after* the device runtime (e.g. SYCL, CUDA) has been already
        // unloaded (the module unloading order is undefined). The resources held by the physical
        // device will be released at process exit anyway, so this intentional "leak" is fine.
        physicalDevice->incRef();

        ctx.physicalDevices.push_back(physicalDevice);
      }
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
        {
        #if defined(OIDN_STATIC_LIB)
          init_device_cpu();
        #else
          modules.load("device_cpu");
        #endif
        }
      #endif
      #if defined(OIDN_DEVICE_SYCL)
        if (getEnvVarOrDefault("OIDN_DEVICE_SYCL", 1))
          modules.load("device_sycl");
      #endif
      #if defined(OIDN_DEVICE_CUDA)
        if (getEnvVarOrDefault("OIDN_DEVICE_CUDA", 1))
          modules.load("device_cuda");
      #endif
      #if defined(OIDN_DEVICE_HIP)
        if (getEnvVarOrDefault("OIDN_DEVICE_HIP", 1))
          modules.load("device_hip");
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