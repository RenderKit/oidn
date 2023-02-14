// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "hip_device.h"

OIDN_NAMESPACE_BEGIN

  class HIPDeviceFactory : public DeviceFactory
  {
  public:
    Ref<Device> newDevice() override
    {
      return makeRef<HIPDevice>();
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_hip)
  {
    if (HIPDevice::isSupported())
      Context::registerDeviceFactory<HIPDeviceFactory>(DeviceType::HIP);
  }

OIDN_NAMESPACE_END