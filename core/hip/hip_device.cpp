// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_device.h"
#include "hip_engine.h"
#include "hip_common.h"

namespace oidn {

  void checkError(hipError_t error)
  {
    if (error == hipSuccess)
      return;

    const char* str = hipGetErrorString(error);
    switch (error)
    {
    case hipErrorMemoryAllocation:
      throw Exception(Error::OutOfMemory, str);
    case hipErrorNoDevice:
    case hipErrorInvalidConfiguration:
    case hipErrorNotSupported:
      throw Exception(Error::UnsupportedHardware, str);
    default:
      throw Exception(Error::Unknown, str);
    }
  }

  bool HIPDevice::isSupported()
  {
    int deviceId = 0;
    if (hipGetDevice(&deviceId) != hipSuccess)
      return false;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, deviceId) != hipSuccess)
      return false;
    return prop.managedMemory;
  }

  HIPDevice::HIPDevice(hipStream_t stream)
    : stream(stream) {}

  void HIPDevice::init()
  {
    int deviceId = 0;
    checkError(hipGetDevice(&deviceId));

    hipDeviceProp_t prop;
    checkError(hipGetDeviceProperties(&prop, deviceId));
    maxWorkGroupSize = prop.maxThreadsPerBlock;
    
    if (isVerbose())
      std::cout << "  Device    : " << prop.name << std::endl;

    // Check required hardware features
    if (!prop.managedMemory)
      throw Exception(Error::UnsupportedHardware, "device does not support managed memory");

    tensorDataType  = DataType::Float16;
    tensorLayout    = TensorLayout::chw;
    weightsLayout   = TensorLayout::oihw;
    //tensorBlockSize = 8; // required by Tensor Core operations

    engine = makeRef<HIPEngine>(this, deviceId, stream);
  }

  Storage HIPDevice::getPointerStorage(const void* ptr)
  {
    hipPointerAttribute_t attrib;
    if (hipPointerGetAttributes(&attrib, ptr) != hipSuccess)
      return Storage::Undefined;

    if (attrib.isManaged)
      return Storage::Managed;

    switch (attrib.memoryType)
    {
    case hipMemoryTypeHost:
      return Storage::Host;
    case hipMemoryTypeDevice:
      return Storage::Device;
    case hipMemoryTypeUnified:
      return Storage::Managed;
    default:
      return Storage::Undefined;
    }
  }

  void HIPDevice::wait()
  {
    engine->wait();
  }
}