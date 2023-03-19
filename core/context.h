// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "module.h"
#include "device_factory.h"
#include <mutex>
#include <map>

OIDN_NAMESPACE_BEGIN

// Global library context
class Context
{
public:
  // Returns the lazily initialized global context. Do *not* call from modules!
  static Context& get();

  template<typename DeviceFactoryT>
  static void registerDeviceType(DeviceType type, const std::vector<Ref<PhysicalDevice>>& physicalDevices)
  {
    if (physicalDevices.empty())
      return;

    Context& ctx = getInstance();
    ctx.deviceFactories[type] = std::unique_ptr<DeviceFactory>(new DeviceFactoryT);
    ctx.physicalDevices.insert(ctx.physicalDevices.end(), physicalDevices.begin(), physicalDevices.end());
  }

  bool isDeviceSupported(DeviceType type) const;
  DeviceFactory* getDeviceFactory(DeviceType type) const;
  int getNumPhysicalDevices() const { return static_cast<int>(physicalDevices.size()); }
  const Ref<PhysicalDevice>& getPhysicalDevice(int id) const;

private:
  // Returns the global context without initialization
  static Context& getInstance();

  Context() = default;
  Context(const Context&) = delete;
  Context& operator =(const Context&) = delete;
  void init();

  std::once_flag initFlag;
  ModuleLoader modules;
  std::map<DeviceType, std::unique_ptr<DeviceFactory>> deviceFactories;
  std::vector<Ref<PhysicalDevice>> physicalDevices;
};

OIDN_NAMESPACE_END