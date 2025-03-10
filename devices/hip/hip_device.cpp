// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_device.h"
#include "hip_engine.h"

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

  HIPPhysicalDevice::HIPPhysicalDevice(int deviceID, const hipDeviceProp_t& prop, int score)
    : PhysicalDevice(DeviceType::HIP, score),
      deviceID(deviceID)
  {
    name = HIPDevice::getName(prop);

    hipUUID_t uuid{};
    if (hipDeviceGetUuid(&uuid, deviceID) == hipSuccess)
    {
      memcpy(this->uuid.bytes, uuid.bytes, sizeof(this->uuid.bytes));
      uuidSupported = true;
    }

  #if defined(_WIN32) && HIP_VERSION_MAJOR >= 6
    memcpy(luid.bytes, prop.luid, sizeof(luid.bytes));
    nodeMask = prop.luidDeviceNodeMask;
    luidSupported = true;
  #endif

    pciDomain   = prop.pciDomainID;
    pciBus      = prop.pciBusID;
    pciDevice   = prop.pciDeviceID;
    pciFunction = 0; // implicit
    pciAddressSupported = true;
  }

  std::vector<Ref<PhysicalDevice>> HIPDevice::getPhysicalDevices()
  {
    int numDevices = 0;
    if (hipGetDeviceCount(&numDevices) != hipSuccess)
      return {};

    std::vector<Ref<PhysicalDevice>> devices;
    for (int deviceID = 0; deviceID < numDevices; ++deviceID)
    {
      hipDeviceProp_t prop{};
      if (hipGetDeviceProperties(&prop, deviceID) != hipSuccess)
        continue;

      HIPArch arch = getArch(prop);
      bool isSupported = arch != HIPArch::Unknown;

      if (isSupported)
      {
        int score = (18 << 16) - 1 - deviceID;
        devices.push_back(makeRef<HIPPhysicalDevice>(deviceID, prop, score));
      }
    }

    return devices;
  }

  std::string HIPDevice::getName(const hipDeviceProp_t& prop)
  {
    return strlen(prop.name) > 0 ? prop.name : prop.gcnArchName;
  }

  std::string HIPDevice::getArchName(const hipDeviceProp_t& prop)
  {
    const std::string fullName = prop.gcnArchName;
    const std::string name = fullName.substr(0, fullName.find(':'));

    if (name == "10.3.0 Sienna_Cichlid 18")
      return "gfx1030";
    else
      return name;
  }

  HIPArch HIPDevice::getArch(const hipDeviceProp_t& prop)
  {
    const std::string name = getArchName(prop);

    if (name == "gfx1030")
      return HIPArch::DL;
    if (name == "gfx1100" || name == "gfx1101" || name == "gfx1102" ||
        name == "gfx1200" || name == "gfx1201")
      return HIPArch::WMMA;
    else
      return HIPArch::Unknown;
  }

  bool HIPDevice::isSupported(int deviceID)
  {
    hipDeviceProp_t prop{};
    if (hipGetDeviceProperties(&prop, deviceID) != hipSuccess)
      return false;
    return getArch(prop) != HIPArch::Unknown;
  }

  HIPDevice::HIPDevice(int deviceID, hipStream_t stream)
    : deviceID(deviceID),
      stream(stream)
  {}

  HIPDevice::HIPDevice(const Ref<HIPPhysicalDevice>& physicalDevice)
    : deviceID(physicalDevice->deviceID)
  {}

  HIPDevice::~HIPDevice()
  {
    // We *must* free all HIP resources inside an enter/leave block
    try
    {
      enter();
      subdevices.clear();
      leave();
    }
    catch (...) {}
  }

  void HIPDevice::enter()
  {
    // Save the current HIP device and switch to ours
    if (prevDeviceID >= 0)
    {
      checkError(hipGetDevice(&prevDeviceID));
      if (deviceID != prevDeviceID)
        checkError(hipSetDevice(deviceID));
    }

    // Clear the HIP error state
    hipGetLastError();
  }

  void HIPDevice::leave()
  {
    // Restore the previous HIP device
    if (prevDeviceID >= 0 && deviceID != prevDeviceID)
      checkError(hipSetDevice(prevDeviceID));
  }

  void HIPDevice::init()
  {
    hipDeviceProp_t prop{};
    checkError(hipGetDeviceProperties(&prop, deviceID));
    arch = getArch(prop);
    maxWorkGroupSize = prop.maxThreadsPerBlock;
    subgroupSize = prop.warpSize;

    if (arch == HIPArch::Unknown)
      throw Exception(Error::UnsupportedHardware, "unsupported HIP device architecture");

    // Print device info
    if (isVerbose())
    {
      std::cout << "  Device    : " << getName(prop) << std::endl;
      std::cout << "    Type    : HIP" << std::endl;
      std::cout << "    Arch    : " << getArchName(prop) << std::endl;
      std::cout << "    CUs     : " << prop.multiProcessorCount << std::endl;
    }

    // Save the current HIP device and switch to ours
    checkError(hipGetDevice(&prevDeviceID));
    if (deviceID != prevDeviceID)
      checkError(hipSetDevice(deviceID));

    // Set device properties
    tensorDataType = DataType::Float16;
    weightDataType = DataType::Float16;
    tensorLayout   = TensorLayout::hwc;
    weightLayout   = TensorLayout::ohwi;
    tensorBlockC   = (arch == HIPArch::DL) ? 32 : 8;

    managedMemorySupported = prop.managedMemory;

  #if defined(_WIN32)
    externalMemoryTypes = ExternalMemoryTypeFlag::OpaqueWin32 |
                          ExternalMemoryTypeFlag::OpaqueWin32KMT |
                          ExternalMemoryTypeFlag::D3D11Texture |
                          ExternalMemoryTypeFlag::D3D11TextureKMT |
                          ExternalMemoryTypeFlag::D3D11Resource |
                          ExternalMemoryTypeFlag::D3D11ResourceKMT |
                          ExternalMemoryTypeFlag::D3D12Heap |
                          ExternalMemoryTypeFlag::D3D12Resource;
  #else
    externalMemoryTypes = ExternalMemoryTypeFlag::OpaqueFD;
  #endif

    subdevices.emplace_back(new Subdevice(std::unique_ptr<Engine>(new HIPEngine(this, stream))));
  }

  Storage HIPDevice::getPtrStorage(const void* ptr)
  {
    hipPointerAttribute_t attrib;
    if (hipPointerGetAttributes(&attrib, ptr) != hipSuccess)
      return Storage::Undefined;

  #if HIP_VERSION_MAJOR >= 6
    switch (attrib.type)
  #else
    switch (attrib.memoryType)
  #endif
    {
    case hipMemoryTypeHost:
      return Storage::Host;
    case hipMemoryTypeDevice:
      return Storage::Device;
    case hipMemoryTypeManaged:
      return Storage::Managed;
    default:
      return Storage::Undefined;
    }
  }

  void HIPDevice::wait()
  {
    for (auto& subdevice : subdevices)
      subdevice->getEngine()->wait();
  }

OIDN_NAMESPACE_END