// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "context.h"

namespace oidn {

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
      modules.load("device_cpu");
    #if !defined(__APPLE__)
      modules.load("device_sycl");
      modules.load("device_cuda");
      modules.load("device_hip");
    #endif
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

} // namespace oidn