// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "sycl_device.h"

OIDN_NAMESPACE_BEGIN

  class SYCLDeviceFactory : public SYCLDeviceFactoryBase
  {
  public:
    Ref<Device> newDevice() override
    {
      return makeRef<SYCLDevice>();
    }

    Ref<Device> newDevice(const sycl::queue* queues, int numQueues) override
    {
      if (numQueues < 0)
        throw Exception(Error::InvalidArgument, "invalid number of queues");
      return makeRef<SYCLDevice>(std::vector<sycl::queue>{queues, queues + numQueues});
    }

    Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) override
    {
      assert(physicalDevice->type == DeviceType::SYCL);
      return makeRef<SYCLDevice>(staticRefCast<SYCLPhysicalDevice>(physicalDevice));
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_sycl)
  {
    Context::registerDeviceType<SYCLDeviceFactory>(DeviceType::SYCL, SYCLDevice::getPhysicalDevices());
  }

OIDN_NAMESPACE_END