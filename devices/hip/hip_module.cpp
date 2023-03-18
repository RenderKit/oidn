// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "hip_device.h"

OIDN_NAMESPACE_BEGIN

  class HIPDeviceFactory : public HIPDeviceFactoryBase
  {
  public:
    Ref<Device> newDevice() override
    {
      return makeRef<HIPDevice>();
    }

    Ref<Device> newDevice(const int* deviceIds, const hipStream_t* streams, int num) override
    {
      if (num != 1)
        throw Exception(Error::InvalidArgument, "invalid number of HIP devices/streams");
      return makeRef<HIPDevice>(deviceIds[0], streams[0]);
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_hip)
  {
    if (HIPDevice::isSupported())
      Context::registerDeviceFactory<HIPDeviceFactory>(DeviceType::HIP);
  }

OIDN_NAMESPACE_END