// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include <map>

OIDN_NAMESPACE_BEGIN

  class DeviceFactory : public RefCount
  {
  public:
    virtual Ref<Device> newDevice(const Ref<PhysicalDevice>& physicalDevice) = 0;
  };

  class SYCLDeviceFactoryBase : public DeviceFactory
  {
  public:
    using DeviceFactory::newDevice;

    virtual bool isDeviceSupported(const sycl::device* device) = 0;
    virtual Ref<Device> newDevice(const sycl::queue* queues, int numQueues) = 0;
  };

  class CUDADeviceFactoryBase : public DeviceFactory
  {
  public:
    using DeviceFactory::newDevice;

    virtual bool isDeviceSupported(int deviceID) = 0;
    virtual Ref<Device> newDevice(const int* deviceIDs, const cudaStream_t* streams, int numPairs) = 0;
  };

  class HIPDeviceFactoryBase : public DeviceFactory
  {
  public:
    using DeviceFactory::newDevice;

    virtual bool isDeviceSupported(int deviceID) = 0;
    virtual Ref<Device> newDevice(const int* deviceIDs, const hipStream_t* streams, int numPairs) = 0;
  };

  class MetalDeviceFactoryBase : public DeviceFactory
  {
  public:
    using DeviceFactory::newDevice;

    virtual bool isDeviceSupported(MTLDevice_id device) = 0;
    virtual Ref<Device> newDevice(const MTLCommandQueue_id* commandQueues, int numQueues) = 0;
  };

OIDN_NAMESPACE_END