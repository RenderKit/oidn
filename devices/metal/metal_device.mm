// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_device.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  MetalPhysicalDevice::MetalPhysicalDevice(id<MTLDevice> device, int score)
    : PhysicalDevice(DeviceType::Metal, score),
      device(device)
  {
    name = [[device name] UTF8String];
  }

  std::vector<Ref<PhysicalDevice>> MetalDevice::getPhysicalDevices()
  {
    @autoreleasepool
    {
      std::vector<Ref<PhysicalDevice>> physicalDevices;
      NSArray* devices = [MTLCopyAllDevices() autorelease];
      const int numDevices = static_cast<int>([devices count]);
      for (int deviceID = 0; deviceID < numDevices; ++deviceID)
      {
        id<MTLDevice> device = devices[deviceID];
        const int score = (2 << 16) - 1 - deviceID;
        physicalDevices.push_back(makeRef<MetalPhysicalDevice>(device, score));
      }
      return physicalDevices;
    }
  }

  MetalDevice::MetalDevice()
  {
    @autoreleasepool
    {
      device = MTLCreateSystemDefaultDevice();
      if (!device)
        throw Exception(Error::UnsupportedHardware, "could not create default Metal device");
    }
  }

  MetalDevice::MetalDevice(const Ref<MetalPhysicalDevice>& physicalDevice)
    : device(physicalDevice->device)
  {}

  MetalDevice::~MetalDevice()
  {
    [device release];
  }

  void MetalDevice::init()
  {
    // Print device info
    if (isVerbose())
    {
      const std::string name = [[device name] UTF8String];

      std::cout << "  Device    : " << name << std::endl;
      std::cout << "    Type    : Metal" << std::endl;
    }

    // Set device properties
    tensorLayout = TensorLayout::hwc;
    weightLayout = TensorLayout::oihw;
    tensorBlockC = 1;

    systemMemorySupported  = false;
    managedMemorySupported = false;

    @autoreleasepool
    {
      engine = makeRef<MetalEngine>(this);
    }
  }

  Storage MetalDevice::getPtrStorage(const void* ptr)
  {
    // USM not supported by Metal
    return Storage::Undefined;
  }

  void MetalDevice::wait()
  {
    if (engine)
      engine->wait();
  }

OIDN_NAMESPACE_END
