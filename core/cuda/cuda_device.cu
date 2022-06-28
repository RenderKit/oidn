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

  bool CUDADevice::isSupported()
  {
    int deviceId = 0;
    if (cudaGetDevice(&deviceId) != cudaSuccess)
      return false;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, deviceId) != cudaSuccess)
      return false;
    const int computeCapability = prop.major * 10 + prop.minor;
    return computeCapability >= minComputeCapability && computeCapability <= maxComputeCapability &&
           prop.unifiedAddressing && prop.managedMemory;
  }

  CUDADevice::CUDADevice(cudaStream_t stream)
    : stream(stream) {}

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

  void CUDADevice::init()
  {
    checkError(cudaGetDevice(&deviceId));

    cudaDeviceProp prop;
    checkError(cudaGetDeviceProperties(&prop, deviceId));
    maxWorkGroupSize = prop.maxThreadsPerBlock;
    computeCapability = prop.major * 10 + prop.minor;

    if (isVerbose())
      std::cout << "  Device  : " << prop.name << std::endl;

    // Check required hardware features
    if (computeCapability < minComputeCapability || computeCapability > maxComputeCapability)
      throw Exception(Error::UnsupportedHardware, "device has unsupported compute capability");
    if (!prop.unifiedAddressing)
      throw Exception(Error::UnsupportedHardware, "device does not support unified addressing");
    if (!prop.managedMemory)
      throw Exception(Error::UnsupportedHardware, "device does not support managed memory");

    tensorDataType  = DataType::Float16;
    tensorLayout    = TensorLayout::hwc;
    weightsLayout   = TensorLayout::ohwi;
    tensorBlockSize = 8; // required by Tensor Core operations
  }

  void CUDADevice::wait()
  {
    checkError(cudaStreamSynchronize(stream));
  }

  std::shared_ptr<Conv> CUDADevice::newConv(const ConvDesc& desc)
  {
    return newCUDAConv(this, desc);
  }

  std::shared_ptr<ConcatConv> CUDADevice::newConcatConv(const ConcatConvDesc& desc)
  {
    if (tensorLayout == TensorLayout::hwc)
      return std::make_shared<CUDAConcatConv>(this, desc);
    else
      return std::make_shared<CHWConcatConv>(this, desc);
  }

  std::shared_ptr<Pool> CUDADevice::newPool(const PoolDesc& desc)
  {
    return std::make_shared<GPUPool<CUDADevice, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<Upsample> CUDADevice::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<GPUUpsample<CUDADevice, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<Autoexposure> CUDADevice::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<GPUAutoexposure<CUDADevice, 1024>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> CUDADevice::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<CUDADevice, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<OutputProcess> CUDADevice::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<CUDADevice, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<ImageCopy> CUDADevice::newImageCopy()
  {
    return std::make_shared<GPUImageCopy<CUDADevice>>(this);
  }

  void* CUDADevice::malloc(size_t byteSize, Storage storage)
  {
    void* ptr;

    switch (storage)
    {
    case Storage::Undefined:
    case Storage::Host:
      checkError(cudaMallocHost(&ptr, byteSize));
      return ptr;

    case Storage::Device:
      checkError(cudaMalloc(&ptr, byteSize));
      return ptr;

    case Storage::Managed:
      checkError(cudaMallocManaged(&ptr, byteSize));
      return ptr;

    default:
      throw Exception(Error::InvalidArgument, "invalid storage mode");
    }
  }

  void CUDADevice::free(void* ptr, Storage storage)
  {
    if (storage == Storage::Host)
      checkError(cudaFreeHost(ptr));
    else
      checkError(cudaFree(ptr));
  }

  void CUDADevice::memcpy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    checkError(cudaMemcpy(dstPtr, srcPtr, byteSize, cudaMemcpyDefault));
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
      return attrib.device == deviceId ? Storage::Device : Storage::Undefined;

    case cudaMemoryTypeManaged:
      return Storage::Managed;

    default:
      return Storage::Undefined;
    }
  }

  namespace
  {
    void CUDART_CB hostFuncCallback(cudaStream_t stream, cudaError_t status, void* fPtr)
    {
      std::unique_ptr<std::function<void()>> f(reinterpret_cast<std::function<void()>*>(fPtr));
      if (status == cudaSuccess)
        (*f)();
    }
  }

  void CUDADevice::runHostFuncAsync(std::function<void()>&& f)
  {
    auto fPtr = new std::function<void()>(std::move(f));
    checkError(cudaStreamAddCallback(stream, hostFuncCallback, fPtr, 0));
  }
}