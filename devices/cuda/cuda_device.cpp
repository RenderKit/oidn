// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_device.h"
#include "cuda_engine.h"
#include "core/subdevice.h"

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_DEVICE_CUDA_API_DRIVER)
  void checkError(CUresult result)
  {
    if (result == CUDA_SUCCESS)
      return;

    const char* str = "";
    if (cuGetErrorString(result, &str) != CUDA_SUCCESS)
      str = "unknown CUDA error";

    switch (result)
    {
    case CUDA_ERROR_OUT_OF_MEMORY:
      throw Exception(Error::OutOfMemory, str);
    case CUDA_ERROR_NO_DEVICE:
    case CUDA_ERROR_INVALID_DEVICE:
    case CUDA_ERROR_NOT_SUPPORTED:
      throw Exception(Error::UnsupportedHardware, str);
    default:
      throw Exception(Error::Unknown, str);
    }
  }
#endif

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

      if (isSupported(prop))
      {
        int score = (19 << 16) - 1 - deviceID;
        devices.push_back(makeRef<CUDAPhysicalDevice>(deviceID, prop, score));
      }
    }

    return devices;
  }

  bool CUDADevice::isSupported(const cudaDeviceProp& prop)
  {
    const int smArch = prop.major * 10 + prop.minor;
    return ((smArch >= 70 && smArch <= 109) || (smArch >= 120 && smArch <= 129)) &&
           prop.unifiedAddressing;
  }

  bool CUDADevice::isSupported(int deviceID)
  {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, deviceID) != cudaSuccess)
      return false;
    return isSupported(prop);
  }

  CUDADevice::CUDADevice(int deviceID, cudaStream_t stream)
    : deviceID(deviceID),
      stream(stream)
  {}

  CUDADevice::CUDADevice(const Ref<CUDAPhysicalDevice>& physicalDevice)
    : deviceID(physicalDevice->deviceID)
  {}

  CUDADevice::~CUDADevice()
  {
    // We *must* free all CUDA resources inside an enter/leave block
    try
    {
      enter();
      subdevices.clear();

    #if defined(OIDN_DEVICE_CUDA_API_DRIVER)
      if (context)
      {
        // Release the CUDA context
        curtn::cleanupContext();
        cuDevicePrimaryCtxRelease(deviceHandle);
      }
    #endif

      leave();
    }
    catch (...) {}
  }

  void CUDADevice::enter()
  {
  #if defined(OIDN_DEVICE_CUDA_API_DRIVER)
    // Save the current CUDA context and switch to ours
    if (context)
      checkError(cuCtxPushCurrent(context));
  #else
    // Save the current CUDA device and switch to ours
    if (prevDeviceID >= 0)
    {
      checkError(cudaGetDevice(&prevDeviceID));
      if (deviceID != prevDeviceID)
        checkError(cudaSetDevice(deviceID));
    }
  #endif

    // Clear the CUDA error state
    cudaGetLastError();
  }

  void CUDADevice::leave()
  {
  #if defined(OIDN_DEVICE_CUDA_API_DRIVER)
    // Restore the previous CUDA context
    if (context)
      checkError(cuCtxPopCurrent(nullptr));
  #else
    // Restore the previous CUDA device
    if (prevDeviceID >= 0 && deviceID != prevDeviceID)
      checkError(cudaSetDevice(prevDeviceID));
  #endif
  }

  void CUDADevice::init()
  {
    cudaDeviceProp prop{};
    checkError(cudaGetDeviceProperties(&prop, deviceID));
    maxWorkGroupSize = prop.maxThreadsPerBlock;
    subgroupSize = prop.warpSize;
    smArch = prop.major * 10 + prop.minor;

    // Check required hardware features
    if (!isSupported(prop))
      throw Exception(Error::UnsupportedHardware, "unsupported CUDA device");

    // Print device info
    if (isVerbose())
    {
      std::cout << "  Device    : " << prop.name << std::endl;
      std::cout << "    Type    : CUDA" << std::endl;
      std::cout << "    Arch    : SM " << prop.major << "." << prop.minor << std::endl;
      std::cout << "    SMs     : " << prop.multiProcessorCount << std::endl;
    }

  #if defined(OIDN_DEVICE_CUDA_API_DRIVER)
    // Initialize the CUDA context and make it current
    checkError(cuDeviceGet(&deviceHandle, deviceID));
    checkError(cuDevicePrimaryCtxRetain(&context, deviceHandle));
    checkError(cuCtxPushCurrent(context)); // between enter/leave, context will be popped in leave()
    checkError(curtn::initContext());
  #else
    // Save the current CUDA device and switch to ours
    checkError(cudaGetDevice(&prevDeviceID));
    if (deviceID != prevDeviceID)
      checkError(cudaSetDevice(deviceID));
  #endif

    // Set device properties
    tensorDataType = DataType::Float16;
    weightDataType = DataType::Float16;
    tensorLayout   = TensorLayout::hwc;
    weightLayout   = TensorLayout::ohwi;
    tensorBlockC   = 8; // required by Tensor Core operations

    systemMemorySupported  = prop.pageableMemoryAccess;
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

    subdevices.emplace_back(new Subdevice(std::unique_ptr<Engine>(new CUDAEngine(this, stream))));
  }

  Storage CUDADevice::getPtrStorage(const void* ptr)
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
      return systemMemorySupported ? Storage::Managed : Storage::Undefined;
    }
  }

  void CUDADevice::wait()
  {
    for (auto& subdevice : subdevices)
      subdevice->getEngine()->wait();
  }

OIDN_NAMESPACE_END