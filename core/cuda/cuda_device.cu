// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_upsample.h"
#include "../gpu/gpu_image_copy.h"
#include "cuda_device.h"
#include "cuda_common.h"
#include "cuda_conv.h"
#include "cuda_concat_conv.h"
#include "cuda_pool.h"

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

  CUDADevice::~CUDADevice()
  {
    checkError(cudnnDestroy(cudnnHandle));
  }

  void CUDADevice::init()
  {
    checkError(cudnnCreate(&cudnnHandle));

    tensorDataType  = DataType::Float16;
    tensorLayout    = TensorLayout::hwc;
    weightsLayout   = TensorLayout::ohwi;
    tensorBlockSize = 8; // required by Tensor Core operations
  }

  void CUDADevice::wait()
  {
    checkError(cudaDeviceSynchronize());
  }

  void CUDADevice::printInfo()
  {
    cudaDeviceProp prop;
    checkError(cudaGetDeviceProperties(&prop, 0));

    std::cout << "  Device  : " << prop.name << std::endl;
  }

  std::shared_ptr<Conv> CUDADevice::newConv(const ConvDesc& desc)
  {
    return std::make_shared<CUDAConv>(this, desc);
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
    return std::make_shared<CUDAPool>(this, desc);
  }

  std::shared_ptr<Upsample> CUDADevice::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<GPUUpsample<CUDAOp, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<Autoexposure> CUDADevice::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<GPUAutoexposure<CUDAOp>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> CUDADevice::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<CUDAOp, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<OutputProcess> CUDADevice::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<CUDAOp, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<ImageCopy> CUDADevice::newImageCopy()
  {
    return std::make_shared<GPUImageCopy<CUDAOp>>(this);
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
}