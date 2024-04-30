// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/context.h"
#include "cuda_device.h"

OIDN_NAMESPACE_BEGIN

  class CUDADeviceFactory : public CUDADeviceFactoryBase
  {
  public:
    bool isDeviceSupported(int deviceID) override
    {
      return CUDADevice::isSupported(deviceID);
    }

    Ref<Device> newDevice(const int* deviceIDs, const cudaStream_t* streams, int numPairs) override
    {
      if (numPairs != 1)
        throw Exception(Error::InvalidArgument, "invalid number of CUDA devices/streams");
      if (deviceIDs == nullptr)
        throw Exception(Error::InvalidArgument, "array of CUDA devices is null");
      if (streams == nullptr)
        throw Exception(Error::InvalidArgument, "array of CUDA streams is null");

      return makeRef<CUDADevice>(deviceIDs[0], streams[0]);
    }

    Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) override
    {
      assert(physicalDevice->type == DeviceType::CUDA);
      return makeRef<CUDADevice>(staticRefCast<CUDAPhysicalDevice>(physicalDevice));
    }
  };

  OIDN_DECLARE_INIT_MODULE(device_cuda)
  {
  #if defined(OIDN_DEVICE_CUDA_API_DRIVER)
    if (curtn::init() != cudaSuccess)
      return;
  #endif

    Context::registerDeviceType<CUDADeviceFactory>(DeviceType::CUDA, CUDADevice::getPhysicalDevices());
  }

OIDN_NAMESPACE_END