// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "context.h"

OIDN_NAMESPACE_BEGIN

  Context& Context::get()
  {
    Context& ctx = getInstance();
    ctx.init();
    return ctx;
  }

  Context& Context::getInstance()
  {
    static Context instance;
    return instance;
  }

  void Context::init()
  {
    std::call_once(initFlag, [this]()
    {
      // Load the modules
    #if defined(OIDN_DEVICE_CPU)
      modules.load("device_cpu");
    #endif
    #if defined(OIDN_DEVICE_SYCL)
      modules.load("device_sycl");
    #endif
    #if defined(OIDN_DEVICE_CUDA)
      modules.load("device_cuda");
    #endif
    #if defined(OIDN_DEVICE_HIP)
      modules.load("device_hip");
    #endif

      // Sort the physical devices by score
      std::sort(physicalDevices.begin(), physicalDevices.end(),
                [](const Ref<PhysicalDevice>& a, const Ref<PhysicalDevice>& b)
                { return a->score > b->score; });
    });
  }

  bool Context::isDeviceSupported(DeviceType type) const
  {
    return deviceFactories.find(type) != deviceFactories.end();
  }

  DeviceFactory* Context::getDeviceFactory(DeviceType type) const
  {
    auto it = deviceFactories.find(type);
    if (it == deviceFactories.end())
      throw Exception(Error::UnsupportedHardware, "unsupported device type");
    return it->second.get();
  }

  const Ref<PhysicalDevice>& Context::getPhysicalDevice(int id) const
  {
    if (id < 0 || static_cast<size_t>(id) >= physicalDevices.size())
      throw Exception(Error::InvalidArgument, "invalid physical device ID");
    return physicalDevices[id];
  }

  Ref<Device> Context::newDevice(int physicalDeviceID)
  {
    const auto& physicalDevice = getPhysicalDevice(physicalDeviceID);
    const DeviceType type = physicalDevice->type;
    return getDeviceFactory(type)->newDevice(physicalDevice);
  }

OIDN_NAMESPACE_END