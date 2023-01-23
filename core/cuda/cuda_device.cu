// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_device.h"
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_pool.h"
#include "../gpu/gpu_upsample.h"
#include "../gpu/gpu_image_copy.h"
#include "cuda_conv.h"
#include "cuda_concat_conv.h"

namespace oidn {

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

  bool CUDADevice::isSupported()
  {
    int deviceId = 0;
    if (cudaGetDevice(&deviceId) != cudaSuccess)
      return false;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess)
      return false;
    const int smArch = prop.major * 10 + prop.minor;
    return smArch >= minSMArch && smArch <= maxSMArch &&
           prop.unifiedAddressing && prop.managedMemory;
  }

  CUDADevice::CUDADevice(cudaStream_t stream)
    : stream(stream) {}

  void CUDADevice::init()
  {
    int deviceId = 0;
    checkError(cudaGetDevice(&deviceId));

    cudaDeviceProp prop;
    checkError(cudaGetDeviceProperties(&prop, deviceId));
    maxWorkGroupSize = prop.maxThreadsPerBlock;
    smArch = prop.major * 10 + prop.minor;

    if (isVerbose())
    {
      std::cout << "  Device    : " << prop.name << std::endl;
      std::cout << "    Arch    : SM " << prop.major << "." << prop.minor << std::endl;
      std::cout << "    SMs     : " << prop.multiProcessorCount << std::endl;
    }

    // Check required hardware features
    if (smArch < minSMArch || smArch > maxSMArch)
      throw Exception(Error::UnsupportedHardware, "device has unsupported compute capability");
    if (!prop.unifiedAddressing)
      throw Exception(Error::UnsupportedHardware, "device does not support unified addressing");
    if (!prop.managedMemory)
      throw Exception(Error::UnsupportedHardware, "device does not support managed memory");

    tensorDataType = DataType::Float16;
    tensorLayout   = TensorLayout::hwc;
    weightsLayout  = TensorLayout::ohwi;
    tensorBlockC   = 8; // required by Tensor Core operations

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

    engine = makeRef<CUDAEngine>(this, deviceId, stream);
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
    engine->wait();
  }
}