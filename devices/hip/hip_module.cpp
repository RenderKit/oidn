// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "hip_device.h"

OIDN_NAMESPACE_BEGIN

  class HIPDeviceFactory : public HIPDeviceFactoryBase
  {
  public:
    bool isDeviceSupported(int deviceID) override
    {
      return HIPDevice::isSupported(deviceID);
    }

    Ref<Device> newDevice(const int* deviceIDs, const hipStream_t* streams, int numPairs) override
    {
      if (numPairs != 1)
        throw Exception(Error::InvalidArgument, "invalid number of HIP devices/streams");
      if (deviceIDs == nullptr)
        throw Exception(Error::InvalidArgument, "array of HIP devices is null");
      if (streams == nullptr)
        throw Exception(Error::InvalidArgument, "array of HIP streams is null");

      return makeRef<HIPDevice>(deviceIDs[0], streams[0]);
    }

    Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) override
    {
      assert(physicalDevice->type == DeviceType::HIP);
      return makeRef<HIPDevice>(staticRefCast<HIPPhysicalDevice>(physicalDevice));
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_hip)
  {
    Context::registerDeviceType<HIPDeviceFactory>(DeviceType::HIP, HIPDevice::getPhysicalDevices());
  }

OIDN_NAMESPACE_END