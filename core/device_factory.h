// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include <map>

namespace oidn {

  class DeviceFactory : public RefCount
  {
  public:
    virtual Ref<Device> newDevice() = 0;
  };

  class SYCLDeviceFactoryBase : public DeviceFactory
  {
  public:
    using DeviceFactory::newDevice;
    
    virtual Ref<Device> newDevice(const sycl::queue* queues, int numQueues) = 0;
  };

} // namespace oidn