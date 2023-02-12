// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../context.h"
#include "cpu_device.h"

OIDN_NAMESPACE_BEGIN

  class CPUDeviceFactory : public DeviceFactory
  {
  public:
    Ref<Device> newDevice() override
    {
      return makeRef<CPUDevice>();
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_cpu)
  {
    if (CPUDevice::isSupported())
      Context::registerDeviceFactory<CPUDeviceFactory>(DeviceType::CPU);
  }

OIDN_NAMESPACE_END