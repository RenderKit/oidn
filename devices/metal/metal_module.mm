// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "metal_device.h"

OIDN_NAMESPACE_BEGIN

  class MetalDeviceFactory : public DeviceFactory
  {
  public:
    Ref<Device> newDevice() override
    {
      return makeRef<MetalDevice>();
    }

    Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) override
    {
      assert(physicalDevice->type == DeviceType::Metal);
      return makeRef<MetalDevice>(staticRefCast<MetalPhysicalDevice>(physicalDevice));
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_metal)
  {
    Context::registerDeviceType<MetalDeviceFactory>(DeviceType::Metal, MetalDevice::getPhysicalDevices());
  }

OIDN_NAMESPACE_END
