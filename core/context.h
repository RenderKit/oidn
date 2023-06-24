// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module.h"
#include "device_factory.h"
#include <mutex>
#include <map>

OIDN_NAMESPACE_BEGIN

// Global library context
class Context : public Verbose
{
public:
  // Returns the global context without initialization
  static Context& get();

  // Initializes the global context (should be called by API functions)
  void init();

  template<typename DeviceFactoryT>
  static void registerDeviceType(DeviceType type, const std::vector<Ref<PhysicalDevice>>& physicalDevices)
  {
    if (physicalDevices.empty())
      return;

    Context& ctx = get();
    ctx.deviceFactories[type] = std::unique_ptr<DeviceFactory>(new DeviceFactoryT);
    ctx.physicalDevices.insert(ctx.physicalDevices.end(), physicalDevices.begin(), physicalDevices.end());
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