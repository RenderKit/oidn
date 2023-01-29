// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../context.h"
#include "cuda_device.h"

namespace oidn {

  class CUDADeviceFactory : public DeviceFactory
  {
  public:
    Ref<Device> newDevice() override
    {
      return makeRef<CUDADevice>();
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_cuda)
  {
    if (CUDADevice::isSupported())
      Context::registerDeviceFactory<CUDADeviceFactory>(DeviceType::CUDA);
  }

} // namespace oidn