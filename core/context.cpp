// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "context.h"

OIDN_NAMESPACE_BEGIN

  Context& Context::get()
  {
    static Context instance;
    return instance;
  }

  bool Context::isDeviceSupported(DeviceType type) const
  {
    return deviceFactories.find(type) != deviceFactories.end();
  }

  DeviceFactory* Context::getDeviceFactory(DeviceType type) const
  {
    auto it = deviceFactories.find(type);
    if (it == deviceFactories.end())
      throw Exception(Error::UnsupportedHardware, "unsupported device type: " + toString(type));
    return it->second.get();
  }

  const Ref<PhysicalDevice>& Context::getPhysicalDevice(int id) const
  {
    if (id < 0 || static_cast<size_t>(id) >= physicalDevices.size())
      throw Exception(Error::InvalidArgument, "invalid physical device ID: " + toString(id));
    return physicalDevices[id];
  }

  Ref<Device> Context::newDevice(int physicalDeviceID)
  {
    const auto& physicalDevice = getPhysicalDevice(physicalDeviceID);
    const DeviceType type = physicalDevice->type;
    return getDeviceFactory(type)->newDevice(physicalDevice);
  }

  Ref<Device> Context::newDevice(DeviceType type)
  {
    if (type == DeviceType::Default)
      return newDevice(0);

    // Find the first physical device of the specified type
    for (const auto& physicalDevice : physicalDevices)
    {
      if (physicalDevice->type == type)
        return getDeviceFactory(type)->newDevice(physicalDevice);
    }

    throw Exception(Error::UnsupportedHardware, "unsupported device type: " + toString(type));
  }

OIDN_NAMESPACE_END