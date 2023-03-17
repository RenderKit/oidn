// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "cuda_device.h"

OIDN_NAMESPACE_BEGIN

  class CUDADeviceFactory : public CUDADeviceFactoryBase
  {
  public:
    Ref<Device> newDevice() override
    {
      return makeRef<CUDADevice>();
    }

    Ref<Device> newDevice(const int* deviceIds, const cudaStream_t* streams, int num) override
    {
      if (num != 1)
        throw Exception(Error::InvalidArgument, "invalid number of CUDA devices/streams");
      return makeRef<CUDADevice>(deviceIds[0], streams[0]);
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_cuda)
  {
    if (CUDADevice::isSupported())
      Context::registerDeviceFactory<CUDADeviceFactory>(DeviceType::CUDA);
  }

OIDN_NAMESPACE_END