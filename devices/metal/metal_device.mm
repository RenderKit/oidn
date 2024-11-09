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
    name = device.name.UTF8String;

    // Report registry ID as LUID
    const uint64_t registryID = device.registryID;
    memcpy(luid.bytes, &registryID, sizeof(luid.bytes));
    nodeMask = 1;
    luidSupported = true;
  }

  std::vector<Ref<PhysicalDevice>> MetalDevice::getPhysicalDevices()
  {
    @autoreleasepool
    {
      std::vector<Ref<PhysicalDevice>> physicalDevices;
    #if TARGET_OS_OSX || TARGET_OS_MACCATALYST
      NSArray* devices = [MTLCopyAllDevices() autorelease];
    #else
      NSArray* devices = @[MTLCreateSystemDefaultDevice()];
    #endif
      const int numDevices = static_cast<int>(devices.count);

      for (int deviceID = 0; deviceID < numDevices; ++deviceID)
      {
        id<MTLDevice> device = devices[deviceID];
        if (MetalDevice::isSupported(device))
        {
          const int score = (2 << 16) - 1 - deviceID;
          physicalDevices.push_back(makeRef<MetalPhysicalDevice>(device, score));
        }
      }

      return physicalDevices;
    }
  }

  bool MetalDevice::isSupported(id<MTLDevice> device)
  {
    if (!device)
      throw Exception(Error::InvalidArgument, "Metal device is null");

    if (@available(macOS 13, iOS 16, tvOS 16, *))
    {
      return [device supportsFamily: MTLGPUFamilyMetal3] &&
             [device supportsFamily: MTLGPUFamilyApple6] && // validated only on Apple GPUs
             device.maxThreadsPerThreadgroup.width >= 1024;

    }
    else
      return false;
  }

  MetalDevice::MetalDevice(const Ref<MetalPhysicalDevice>& physicalDevice)
    : device(physicalDevice->device)
  {}

  MetalDevice::MetalDevice(id<MTLCommandQueue> commandQueue)
  {
    if (!commandQueue)
      throw Exception(Error::InvalidArgument, "Metal command queue is null");

    device = commandQueue.device;
    userCommandQueue = commandQueue;
  }

  MetalDevice::~MetalDevice()
  {
    [device release];
  }

  void MetalDevice::init()
  {
    @autoreleasepool
    {
      if (!isSupported(device))
        throw Exception(Error::UnsupportedHardware, "unsupported Metal device");

      // Print device info
      if (isVerbose())
      {
        const std::string name = device.name.UTF8String;

        std::cout << "  Device    : " << name << std::endl;
        std::cout << "    Type    : Metal" << std::endl;
        if (@available(macOS 14, iOS 17, tvOS 17, *))
          std::cout << "    Arch    : " << device.architecture.name.UTF8String << std::endl;
      }

      // Set device properties
      tensorDataType = DataType::Float16;
      weightDataType = DataType::Float16;
      tensorLayout   = TensorLayout::hwc;
      weightLayout   = TensorLayout::oihw;
      tensorBlockC   = 1;

      minTileAlignment = 32; // MPS convolution seems to require this for consistent output

      systemMemorySupported  = false;
      managedMemorySupported = false; // unsupported due to manual synchronization

      subdevices.emplace_back(new Subdevice(std::unique_ptr<Engine>(new MetalEngine(this))));
    }
  }

  Storage MetalDevice::getPtrStorage(const void* ptr)
  {
    // USM not supported by Metal
    return Storage::Undefined;
  }

  void MetalDevice::execute(std::function<void()>&& f, SyncMode sync)
  {
    @autoreleasepool
    {
      Device::execute(std::move(f), sync);
    }
  }

  void MetalDevice::flush()
  {
    for (auto& subdevice : subdevices)
      subdevice->getEngine()->flush();
  }

  void MetalDevice::wait()
  {
    for (auto& subdevice : subdevices)
      subdevice->getEngine()->wait();

    //std::cout << "Metal device memory: " << [engine->getMTLDevice() currentAllocatedSize] << std::endl;
  }

OIDN_NAMESPACE_END
