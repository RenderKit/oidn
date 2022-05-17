// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_device.h"
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_pool.h"
#include "../gpu/gpu_upsample.h"
#include "../gpu/gpu_image_copy.h"
#include "hip_common.h"
#include "hip_conv.h"

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
    checkError(hipGetDevice(&deviceId));

    hipDeviceProp_t prop;
    checkError(hipGetDeviceProperties(&prop, deviceId));
    
    if (isVerbose())
      std::cout << "  Device  : " << prop.name << std::endl;

    // Check required hardware features
    if (!prop.managedMemory)
      throw Exception(Error::UnsupportedHardware, "device does not support managed memory");

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

  std::shared_ptr<Conv> HIPDevice::newConv(const ConvDesc& desc)
  {
    return std::make_shared<HIPConv>(this, desc);
  }

  std::shared_ptr<Pool> HIPDevice::newPool(const PoolDesc& desc)
  {
    return std::make_shared<GPUPool<HIPDevice, half, TensorLayout::chw>>(this, desc);
  }

  std::shared_ptr<Upsample> HIPDevice::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<GPUUpsample<HIPDevice, half, TensorLayout::chw>>(this, desc);
  }

  std::shared_ptr<Autoexposure> HIPDevice::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<GPUAutoexposure<HIPDevice>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> HIPDevice::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<HIPDevice, half, TensorLayout::chw>>(this, desc);
  }

  std::shared_ptr<OutputProcess> HIPDevice::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<HIPDevice, half, TensorLayout::chw>>(this, desc);
  }

  std::shared_ptr<ImageCopy> HIPDevice::newImageCopy()
  {
    return std::make_shared<GPUImageCopy<HIPDevice>>(this);
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
      return attrib.device == deviceId ? Storage::Device : Storage::Undefined;

    case hipMemoryTypeUnified:
      return Storage::Managed;

    default:
      return Storage::Undefined;
    }
  }

  namespace
  {
    void hostFuncCallback(hipStream_t stream, hipError_t status, void* fPtr)
    {
      std::unique_ptr<std::function<void()>> f(reinterpret_cast<std::function<void()>*>(fPtr));
      if (status == hipSuccess)
        (*f)();
    }
  }

  void HIPDevice::runHostFuncAsync(std::function<void()>&& f)
  {
    auto fPtr = new std::function<void()>(std::move(f));
    checkError(hipStreamAddCallback(0, hostFuncCallback, fPtr, 0));
  }
}