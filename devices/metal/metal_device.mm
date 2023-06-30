// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_device.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  MetalPhysicalDevice::MetalPhysicalDevice(int deviceID, std::string name, int score)
    : PhysicalDevice(DeviceType::Metal, score),
      deviceID(deviceID)
  {
    this->name = name;
  }

  std::vector<Ref<PhysicalDevice>> MetalDevice::getPhysicalDevices()
  {
    @autoreleasepool
    {
      std::vector<Ref<PhysicalDevice>> physicalDevices;
      NSArray* devices = [MTLCopyAllDevices() autorelease];
      int numDevice = (int)[devices count];
      for (int deviceID = 0 ; deviceID < numDevice ; deviceID++)
      {
        id<MTLDevice>  device = devices[deviceID];
        int score = (19 << 16) - 1 - deviceID;
        std::string name = std::string([[device name] UTF8String]);
        physicalDevices.push_back(makeRef<MetalPhysicalDevice>(deviceID, name, score));
      }
      return physicalDevices;
    }
  }

  MetalDevice::MetalDevice(int deviceID)
    : deviceID(deviceID)
  {
    @autoreleasepool
    {
      device = mtlDevice(deviceID);
    }
  }

  MetalDevice::MetalDevice(const Ref<MetalPhysicalDevice>& physicalDevice)
    : deviceID(physicalDevice->deviceID)
  {
    @autoreleasepool
    {
      device = mtlDevice(deviceID);
    }
  }

  MetalDevice::~MetalDevice()
  {
  }

  void MetalDevice::init()
  {
    tensorLayout = TensorLayout::hwc;
    weightLayout = TensorLayout::oihw;
    tensorBlockC = 1;

    systemMemorySupported  = true;
    managedMemorySupported = false;

    @autoreleasepool
    {
      engine = makeRef<MetalEngine>(this);
    }
  }

  Storage MetalDevice::getPtrStorage(const void* ptr)
  {
    return Storage::Host;
  }

  int MetalDevice::getInt(const std::string& name)
  {
      // TODO: Need to implement something
      return 0;
  }

  void MetalDevice::setInt(const std::string& name, int value)
  {
    // TODO: Need to implement something
  }

  void MetalDevice::wait()
  {
    if (engine)
      engine->wait();
  }

OIDN_NAMESPACE_END
