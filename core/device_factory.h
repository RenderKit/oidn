// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include <map>

OIDN_NAMESPACE_BEGIN

  class DeviceFactory : public RefCount
  {
  public:
    virtual Ref<Device> newDevice() = 0;
  };

  class SYCLDeviceFactoryBase : public DeviceFactory
  {
  public:
    using DeviceFactory::newDevice;
    virtual Ref<Device> newDevice(const sycl::queue* queues, int numQueues) = 0;
  };

  class CUDADeviceFactoryBase : public DeviceFactory
  {
  public:
    using DeviceFactory::newDevice;
    virtual Ref<Device> newDevice(const int* deviceIds, const cudaStream_t* streams, int num) = 0;
  };

  class HIPDeviceFactoryBase : public DeviceFactory
  {
  public:
    using DeviceFactory::newDevice;
    virtual Ref<Device> newDevice(const int* deviceIds, const hipStream_t* streams, int num) = 0;
  };

OIDN_NAMESPACE_END