// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "metal_device.h"

OIDN_NAMESPACE_BEGIN

  class MetalDeviceFactory : public MetalDeviceFactoryBase
  {
  public:
    bool isDeviceSupported(MTLDevice_id device) override
    {
      return MetalDevice::isSupported(device);
    }

    Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) override
    {
      assert(physicalDevice->type == DeviceType::Metal);
      return makeRef<MetalDevice>(staticRefCast<MetalPhysicalDevice>(physicalDevice));
    }

    Ref<Device> newDevice(const MTLCommandQueue_id* commandQueues, int numQueues) override
    {
      if (numQueues != 1)
        throw Exception(Error::InvalidArgument, "invalid number of Metal command queues");
      return makeRef<MetalDevice>(commandQueues[0]);
    }
  };

  OIDN_DECLARE_INIT_STATIC_MODULE(device_metal)
  {
    Context::registerDeviceType<MetalDeviceFactory>(DeviceType::Metal, MetalDevice::getPhysicalDevices());
  }

OIDN_NAMESPACE_END
