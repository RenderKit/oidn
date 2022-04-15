// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_upsample.h"
#include "../gpu/gpu_image_copy.h"
#include "hip_device.h"
#include "hip_common.h"
#include "hip_conv.h"
#include "hip_pool.h"

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

  HIPDevice::~HIPDevice()
  {
    checkError(miopenDestroy(miopenHandle));
  }

  void HIPDevice::init()
  {
    /*
    int deviceId;
    checkError(hipGetDevice(&deviceId));

    int pi;
    checkError(hipDeviceGetAttribute(&pi, hipDeviceAttributeCanUseHostPointerForRegisteredMem, deviceId));
    std::cout << "hipDeviceAttributeCanUseHostPointerForRegisteredMem: " << pi << std::endl;
    */

    checkError(miopenCreate(&miopenHandle));

    //miopenEnableProfiling(miopenHandle, true);

    tensorDataType  = DataType::Float16;
    tensorLayout    = TensorLayout::chw;
    weightsLayout   = TensorLayout::oihw;
    //tensorBlockSize = 8; // required by Tensor Core operations
  }

  void HIPDevice::wait()
  {
    checkError(hipDeviceSynchronize());
  }

  void HIPDevice::printInfo()
  {
    hipDeviceProp_t prop;
    checkError(hipGetDeviceProperties(&prop, 0));

    std::cout << "  Device  : " << prop.name << std::endl;
  }

  std::shared_ptr<Conv> HIPDevice::newConv(const ConvDesc& desc)
  {
    return std::make_shared<HIPConv>(this, desc);
  }

  std::shared_ptr<Pool> HIPDevice::newPool(const PoolDesc& desc)
  {
    return std::make_shared<HIPPool>(this, desc);
  }

  std::shared_ptr<Upsample> HIPDevice::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<GPUUpsample<HIPOp, half, TensorLayout::chw>>(this, desc);
  }

  std::shared_ptr<Autoexposure> HIPDevice::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<GPUAutoexposure<HIPOp>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> HIPDevice::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<HIPOp, half, TensorLayout::chw>>(this, desc);
  }

  std::shared_ptr<OutputProcess> HIPDevice::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<HIPOp, half, TensorLayout::chw>>(this, desc);
  }

  void HIPDevice::imageCopy(const Image& src, const Image& dst)
  {
    //gpuImageCopy<HIPDevice>(this, src, dst);
  }

  void* HIPDevice::malloc(size_t byteSize, Storage storage)
  {
    void* ptr;

    switch (storage)
    {
    case Storage::Undefined:
    case Storage::Host:
      checkError(hipHostMalloc(&ptr, byteSize));
      return ptr;

    case Storage::Device:
      checkError(hipMalloc(&ptr, byteSize));
      return ptr;

    case Storage::Managed:
      checkError(hipMallocManaged(&ptr, byteSize));
      return ptr;

    default:
      throw Exception(Error::InvalidArgument, "invalid storage mode");
    }
  }

  void HIPDevice::free(void* ptr, Storage storage)
  {
    if (storage == Storage::Host)
      checkError(hipHostFree(ptr));
    else
      checkError(hipFree(ptr));
  }

  void HIPDevice::memcpy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    checkError(hipMemcpy(dstPtr, srcPtr, byteSize, hipMemcpyDefault));
  }
}