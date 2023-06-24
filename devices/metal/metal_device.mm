// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include "metal_device.h"
#include "metal_engine.h"

#include <MetalPerformanceShaders/MetalPerformanceShaders.h>

OIDN_NAMESPACE_BEGIN

  MetalPhysicalDevice::MetalPhysicalDevice(int deviceID, std::string name, int score)
    : PhysicalDevice(DeviceType::METAL, score),
      deviceID(deviceID)
  {
    this->name = name;
  }

  std::vector<Ref<PhysicalDevice>> MetalDevice::getPhysicalDevices()
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

  MetalDevice::MetalDevice(int deviceID)
    : deviceID(deviceID)
  {
    device = mtlDevice(deviceID);
  }

  MetalDevice::MetalDevice(const Ref<MetalPhysicalDevice>& physicalDevice)
    : deviceID(physicalDevice->deviceID)
  {
    device = mtlDevice(deviceID);
  }

  MetalDevice::~MetalDevice()
  {
  }

  void MetalDevice::init()
  {
    managedMemorySupported = false;
    systemMemorySupported = true;
    
    tensorLayout = TensorLayout::hwc;
    weightLayout = TensorLayout::oihw;
    tensorBlockC = 1;

    engine = makeRef<MetalEngine>(this);
  }

  Storage MetalDevice::getPointerStorage(const void* ptr)
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
