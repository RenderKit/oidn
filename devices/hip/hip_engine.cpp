// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_engine.h"
#include "hip_external_buffer.h"
#include "hip_conv.h"
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_pool.h"
#include "../gpu/gpu_upsample.h"
#include "../gpu/gpu_image_copy.h"

OIDN_NAMESPACE_BEGIN

  HIPEngine::HIPEngine(const Ref<HIPDevice>& device,
                       int deviceId,
                       hipStream_t stream)
    : device(device.get()),
      deviceId(deviceId),
      stream(stream) {}

  Ref<Buffer> HIPEngine::newExternalBuffer(ExternalMemoryTypeFlag fdType,
                                           int fd, size_t byteSize)
  {
    return makeRef<HIPExternalBuffer>(this, fdType, fd, byteSize);
  }

  Ref<Buffer> HIPEngine::newExternalBuffer(ExternalMemoryTypeFlag handleType,
                                           void* handle, const void* name, size_t byteSize)
  {
    return makeRef<HIPExternalBuffer>(this, handleType, handle, name, byteSize);
  }

  std::shared_ptr<Conv> HIPEngine::newConv(const ConvDesc& desc)
  {
    switch (device->arch)
    {
    case HIPArch::DL:
      return newHIPConvDL(this, desc);
    case HIPArch::XDL:
      return newHIPConvXDL(this, desc);   
    case HIPArch::WMMA:
      return newHIPConvWMMA(this, desc); 
    default:
      throw std::logic_error("unsupported HIP device architecture");
    }
  }

  std::shared_ptr<Pool> HIPEngine::newPool(const PoolDesc& desc)
  {
    return std::make_shared<GPUPool<HIPEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<Upsample> HIPEngine::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<GPUUpsample<HIPEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<Autoexposure> HIPEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<GPUAutoexposure<HIPEngine, 1024>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> HIPEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<HIPEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<OutputProcess> HIPEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<HIPEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<ImageCopy> HIPEngine::newImageCopy()
  {
    return std::make_shared<GPUImageCopy<HIPEngine>>(this);
  }

  void* HIPEngine::malloc(size_t byteSize, Storage storage)
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

  void HIPEngine::free(void* ptr, Storage storage)
  {
    if (storage == Storage::Host)
      checkError(hipHostFree(ptr));
    else
      checkError(hipFree(ptr));
  }

  void HIPEngine::memcpy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    checkError(hipMemcpy(dstPtr, srcPtr, byteSize, hipMemcpyDefault));
  }

  void HIPEngine::submitMemcpy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    checkError(hipMemcpyAsync(dstPtr, srcPtr, byteSize, hipMemcpyDefault, stream));
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

  void HIPEngine::submitHostFunc(std::function<void()>&& f)
  {
    auto fPtr = new std::function<void()>(std::move(f));
    checkError(hipStreamAddCallback(stream, hostFuncCallback, fPtr, 0));
  }

  void HIPEngine::wait()
  {
    checkError(hipStreamSynchronize(stream));
  }

OIDN_NAMESPACE_END