// Copyright 2022 Intel Corporation
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

  HIPEngine::HIPEngine(HIPDevice* device,
                       hipStream_t stream)
    : device(device),
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

  bool HIPEngine::isSupported(const TensorDesc& desc) const
  {
    // CK tensors must be smaller than 2GB
    return Engine::isSupported(desc) && desc.getByteSize() <= INT32_MAX;
  }

  Ref<Conv> HIPEngine::newConv(const ConvDesc& desc)
  {
    return newHIPConv(this, desc);
  }

  Ref<Pool> HIPEngine::newPool(const PoolDesc& desc)
  {
    return makeRef<GPUPool<HIPEngine, half, TensorLayout::hwc>>(this, desc);
  }

  Ref<Upsample> HIPEngine::newUpsample(const UpsampleDesc& desc)
  {
    return makeRef<GPUUpsample<HIPEngine, half, TensorLayout::hwc>>(this, desc);
  }

  Ref<Autoexposure> HIPEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return makeRef<GPUAutoexposure<HIPEngine, 1024>>(this, srcDesc);
  }

  Ref<InputProcess> HIPEngine::newInputProcess(const InputProcessDesc& desc)
  {
    switch (device->getTensorBlockC())
    {
    case 8:
      return makeRef<GPUInputProcess<HIPEngine, half, TensorLayout::hwc,  8>>(this, desc);
    case 32:
      return makeRef<GPUInputProcess<HIPEngine, half, TensorLayout::hwc, 32>>(this, desc);
    default:
      throw std::logic_error("unexpected tensor block channel size");
    }
  }

  Ref<OutputProcess> HIPEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return makeRef<GPUOutputProcess<HIPEngine, half, TensorLayout::hwc>>(this, desc);
  }

  Ref<ImageCopy> HIPEngine::newImageCopy()
  {
    return makeRef<GPUImageCopy<HIPEngine>>(this);
  }

  void* HIPEngine::usmAlloc(size_t byteSize, Storage storage)
  {
    if (byteSize == 0)
      return nullptr;

    void* ptr = nullptr;

    switch (storage)
    {
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

  void HIPEngine::usmFree(void* ptr, Storage storage)
  {
    if (ptr == nullptr)
      return;

    if (storage == Storage::Host)
      checkError(hipHostFree(ptr));
    else
      checkError(hipFree(ptr));
  }

  void HIPEngine::usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    checkError(hipMemcpy(dstPtr, srcPtr, byteSize, hipMemcpyDefault));
  }

  void HIPEngine::submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
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

  void HIPEngine::submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct)
  {
    auto fPtr = new std::function<void()>(std::move(f));
    checkError(hipStreamAddCallback(stream, hostFuncCallback, fPtr, 0));
  }

  void HIPEngine::wait()
  {
    checkError(hipStreamSynchronize(stream));
  }

OIDN_NAMESPACE_END