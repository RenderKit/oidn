// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "cpu_device.h"

OIDN_NAMESPACE_BEGIN

  class CPUDeviceFactory : public DeviceFactory
  {
  public:
    Ref<Device> newDevice() override
    {
      return makeRef<CPUDevice>();
    }

    Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) override
    {
      assert(physicalDevice->type == DeviceType::CPU);
      return makeRef<CPUDevice>();
    }
  };

#if defined(OIDN_STATIC_LIB)
  void init_device_cpu()
#else
  OIDN_DECLARE_INIT_MODULE(device_cpu)
#endif
  {
    Context::registerDeviceType<CPUDeviceFactory>(DeviceType::CPU, CPUDevice::getPhysicalDevices());
  }

OIDN_NAMESPACE_END