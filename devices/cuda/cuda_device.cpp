// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_device.h"
#include "cuda_engine.h"

OIDN_NAMESPACE_BEGIN

  void checkError(cudaError_t error)
  {
    if (error == cudaSuccess)
      return;

    const char* str = cudaGetErrorString(error);
    switch (error)
    {
    case cudaErrorMemoryAllocation:
      throw Exception(Error::OutOfMemory, str);
    case cudaErrorNoDevice:
    case cudaErrorInvalidConfiguration:
    case cudaErrorNotSupported:
      throw Exception(Error::UnsupportedHardware, str);
    default:
      throw Exception(Error::Unknown, str);
    }
  }

  CUDAPhysicalDevice::CUDAPhysicalDevice(int deviceID, const cudaDeviceProp& prop, int score)
    : PhysicalDevice(DeviceType::CUDA, score),
      deviceID(deviceID)
  {
    name = prop.name;

    memcpy(uuid.bytes, prop.uuid.bytes, sizeof(uuid.bytes));
    uuidSupported = true;

  #if defined(_WIN32)
    if (prop.tccDriver == 0)
    {
      memcpy(luid.bytes, prop.luid, sizeof(luid.bytes));
      nodeMask = prop.luidDeviceNodeMask;
      luidSupported = true;
    }
  #endif

    pciDomain   = prop.pciDomainID;
    pciBus      = prop.pciBusID;
    pciDevice   = prop.pciDeviceID;
    pciFunction = 0; // implicit
    pciAddressSupported = true;
  }

  std::vector<Ref<PhysicalDevice>> CUDADevice::getPhysicalDevices()
  {
    int numDevices = 0;
    if (cudaGetDeviceCount(&numDevices) != cudaSuccess)
      return {};

    std::vector<Ref<PhysicalDevice>> devices;
    for (int deviceID = 0; deviceID < numDevices; ++deviceID)
    {
      cudaDeviceProp prop{};
      if (cudaGetDeviceProperties(&prop, deviceID) != cudaSuccess)
        continue;

      int smArch = prop.major * 10 + prop.minor;
      bool isSupported = smArch >= minSMArch && smArch <= maxSMArch &&
                         prop.unifiedAddressing;

      if (isSupported)
      {
        int score = (19 << 16) - 1 - deviceID;
        devices.push_back(makeRef<CUDAPhysicalDevice>(deviceID, prop, score));
      }
    }

    return devices;
  }

  CUDADevice::CUDADevice(int deviceID, cudaStream_t stream)
    : deviceID(deviceID),
      stream(stream)
  {
    if (deviceID < 0)
      checkError(cudaGetDevice(&this->deviceID));
  }

  CUDADevice::CUDADevice(const Ref<CUDAPhysicalDevice>& physicalDevice)
    : deviceID(physicalDevice->deviceID)
  {}

  CUDADevice::~CUDADevice()
  {
    // Make sure to free up all resources inside a begin/end block
    begin();
    engine = nullptr;
    end();
  }

  void CUDADevice::begin()
  {
    assert(prevDeviceID < 0);

    // Save the current CUDA device
    checkError(cudaGetDevice(&prevDeviceID));

    // Set the current CUDA device
    if (deviceID != prevDeviceID)
      checkError(cudaSetDevice(deviceID));
  }

  void CUDADevice::end()
  {
    assert(prevDeviceID >= 0);

    // Restore the previous CUDA device
    if (deviceID != prevDeviceID)
      checkError(cudaSetDevice(prevDeviceID));
    prevDeviceID = -1;
  }

  void CUDADevice::init()
  {
    cudaDeviceProp prop{};
    checkError(cudaGetDeviceProperties(&prop, deviceID));
    maxWorkGroupSize = prop.maxThreadsPerBlock;
    smArch = prop.major * 10 + prop.minor;

    // Check required hardware features
    if (smArch < minSMArch || smArch > maxSMArch)
      throw Exception(Error::UnsupportedHardware, "device has unsupported compute capability");
    if (!prop.unifiedAddressing)
      throw Exception(Error::UnsupportedHardware, "device does not support unified addressing");

    // Print device info
    if (isVerbose())
    {
      std::cout << "  Device    : " << prop.name << std::endl;
      std::cout << "    Arch    : SM " << prop.major << "." << prop.minor << std::endl;
      std::cout << "    SMs     : " << prop.multiProcessorCount << std::endl;
    }

    // Set device properties
    tensorDataType = DataType::Float16;
    tensorLayout   = TensorLayout::hwc;
    weightLayout   = TensorLayout::ohwi;
    tensorBlockC   = 8; // required by Tensor Core operations

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

    engine = makeRef<CUDAEngine>(this, stream);
  }

  Storage CUDADevice::getPointerStorage(const void* ptr)
  {
    cudaPointerAttributes attrib;
    if (cudaPointerGetAttributes(&attrib, ptr) != cudaSuccess)
      return Storage::Undefined;

    switch (attrib.type)
    {
    case cudaMemoryTypeHost:
      return Storage::Host;
    case cudaMemoryTypeDevice:
      return Storage::Device;
    case cudaMemoryTypeManaged:
      return Storage::Managed;
    default:
      return Storage::Undefined;
    }
  }

  void CUDADevice::wait()
  {
    if (engine)
      engine->wait();
  }

OIDN_NAMESPACE_END