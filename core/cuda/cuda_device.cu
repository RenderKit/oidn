// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../xpu/xpu_input_process.h"
#include "../xpu/xpu_output_process.h"
#include "../xpu/xpu_upsample.h"
#include "../xpu/xpu_image_copy.h"
#include "cuda_device.h"
#include "cuda_buffer.h"
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

  Ref<Buffer> CUDADevice::newBuffer(size_t byteSize, MemoryKind kind)
  {
    return makeRef<CUDABuffer>(this, byteSize, kind);
  }

  Ref<Buffer> CUDADevice::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<CUDABuffer>(this, ptr, byteSize);
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
    return std::make_shared<XPUUpsample<CUDAOp, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<InputProcess> CUDADevice::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<XPUInputProcess<CUDAOp, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<OutputProcess> CUDADevice::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<XPUOutputProcess<CUDAOp, half, TensorLayout::hwc>>(this, desc);
  }

  void CUDADevice::imageCopy(const Image& src, const Image& dst)
  {
    xpuImageCopy<CUDADevice>(this, src, dst);
  }
}