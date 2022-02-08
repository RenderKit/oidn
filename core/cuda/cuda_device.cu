// Copyright 2009-2021 Intel Corporation
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

  CUDADevice::CUDADevice() {}

  void CUDADevice::init()
  {
    checkError(cudnnCreate(&cudnnHandle));

    tensorDataType = DataType::Float16;
    tensorLayout = TensorLayout::hwc;
    weightsLayout = TensorLayout::ohwi;
    tensorBlockSize = 8; // required by Tensor Core operations
  }

  CUDADevice::~CUDADevice()
  {
    checkError(cudnnDestroy(cudnnHandle));
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
    return makeRef<CUDABuffer>(Ref<CUDADevice>(this), byteSize, kind);
  }

  Ref<Buffer> CUDADevice::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<CUDABuffer>(Ref<CUDADevice>(this), ptr, byteSize);
  }

  std::shared_ptr<Conv> CUDADevice::newConv(const ConvDesc& desc)
  {
    return std::make_shared<CUDAConv>(Ref<CUDADevice>(this), desc);
  }

  std::shared_ptr<Pool> CUDADevice::newPool(const PoolDesc& desc)
  {
    return std::make_shared<CUDAPool>(Ref<CUDADevice>(this), desc);
  }

  std::shared_ptr<Upsample> CUDADevice::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<XPUUpsample<CUDAOp, half, TensorLayout::hwc>>(Ref<CUDADevice>(this), desc);
  }

  std::shared_ptr<InputProcess> CUDADevice::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<XPUInputProcess<CUDAOp, half, TensorLayout::hwc>>(Ref<CUDADevice>(this), desc);
  }

  std::shared_ptr<OutputProcess> CUDADevice::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<XPUOutputProcess<CUDAOp, half, TensorLayout::hwc>>(Ref<CUDADevice>(this), desc);
  }

  std::shared_ptr<ConcatConv> CUDADevice::newConcatConv(const ConcatConvDesc& desc)
  {
    return std::make_shared<CUDAConcatConv>(Ref<CUDADevice>(this), desc);
  }

  void CUDADevice::imageCopy(const Image& src, const Image& dst)
  {
    xpuImageCopy(Ref<CUDADevice>(this), src, dst);
  }
}