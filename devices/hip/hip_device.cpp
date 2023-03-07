// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_device.h"
#include "hip_engine.h"
#include "ck/host_utility/device_prop.hpp"

OIDN_NAMESPACE_BEGIN

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

    const std::string archStr = ck::get_device_name();
    return getArch(archStr) != HIPArch::Unknown && prop.managedMemory;
  }

  HIPArch HIPDevice::getArch(const std::string& archStr)
  {
    if (archStr == "gfx908" || archStr == "gfx90a")
      return HIPArch::XDL;
    if (archStr == "gfx1030")
      return HIPArch::DL;
    if (archStr == "gfx1100" || archStr == "gfx1101" || archStr == "gfx1102")
      return HIPArch::WMMA;
    return HIPArch::Unknown;
  }

  HIPDevice::HIPDevice(hipStream_t stream)
    : stream(stream) {}

  void HIPDevice::init()
  {
    int deviceId = 0;
    checkError(hipGetDevice(&deviceId));

    const std::string archStr = ck::get_device_name();
    arch = getArch(archStr);

    hipDeviceProp_t prop;
    checkError(hipGetDeviceProperties(&prop, deviceId));
    maxWorkGroupSize = prop.maxThreadsPerBlock;

    const std::string name = strlen(prop.name) > 0 ? prop.name : "AMD GPU";
    
    if (isVerbose())
    {
      std::cout << "  Device    : " << name << std::endl;
      std::cout << "    Arch    : " << archStr << std::endl;
      std::cout << "    CUs     : " << prop.multiProcessorCount << std::endl;
    }

    if (arch == HIPArch::Unknown)
      throw Exception(Error::UnsupportedHardware, "unsupported HIP device architecture");
    if (!prop.managedMemory)
      throw Exception(Error::UnsupportedHardware, "HIP device does not support managed memory");

    tensorDataType = DataType::Float16;
    tensorLayout   = TensorLayout::hwc;
    weightLayout   = TensorLayout::ohwi;
    tensorBlockC   = (arch == HIPArch::XDL) ? 8 : 32;

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

OIDN_NAMESPACE_END