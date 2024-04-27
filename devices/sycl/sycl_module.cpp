// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "sycl_device.h"

OIDN_NAMESPACE_BEGIN

  class SYCLDeviceFactory : public SYCLDeviceFactoryBase
  {
  public:
    Ref<Device> newDevice(const sycl::queue* queues, int numQueues) override
    {
      if (numQueues < 1)
        throw Exception(Error::InvalidArgument, "invalid number of SYCL queues");
      if (queues == nullptr)
        throw Exception(Error::InvalidArgument, "array of SYCL queues is null");

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
  #if defined(OIDN_DEVICE_SYCL_JIT_CACHE)
    // Enable persistent JIT cache if not disabled explicitly
    setEnvVar("SYCL_CACHE_PERSISTENT", 1, false);
  #endif

    Context::registerDeviceType<SYCLDeviceFactory>(DeviceType::SYCL, SYCLDevice::getPhysicalDevices());
  }

OIDN_NAMESPACE_END