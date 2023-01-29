// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../context.h"
#include "sycl_device.h"

namespace oidn {

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
  };

  OIDN_DECLARE_INIT_MODULE(device_sycl)
  {
    if (SYCLDevice::isSupported())
      Context::registerDeviceFactory<SYCLDeviceFactory>(DeviceType::SYCL);
  }

} // namespace oidn