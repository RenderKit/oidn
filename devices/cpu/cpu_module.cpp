// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "cpu_device.h"

OIDN_NAMESPACE_BEGIN

  class CPUDeviceFactory : public DeviceFactory
  {
  public:
    Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) override
    {
      assert(physicalDevice->type == DeviceType::CPU);
      return makeRef<CPUDevice>();
    }
  };

  OIDN_DECLARE_INIT_STATIC_MODULE(device_cpu)
  {
    Context::registerDeviceType<CPUDeviceFactory>(DeviceType::CPU, CPUDevice::getPhysicalDevices());
  }

OIDN_NAMESPACE_END