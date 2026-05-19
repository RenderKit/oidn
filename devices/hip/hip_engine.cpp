// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_engine.h"
#include "hip_external_buffer.h"
#include "hip_external_semaphore.h"
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

  Ref<Buffer> HIPEngine::newExternalBuffer(ExternalMemoryTypeFlags fdType,
                                           int fd, size_t byteSize)
  {
    return makeRef<HIPExternalBuffer>(this, fdType, fd, byteSize);
  }

  Ref<Buffer> HIPEngine::newExternalBuffer(ExternalMemoryTypeFlags handleType,
                                           void* handle, const void* name, size_t byteSize)
  {
    return makeRef<HIPExternalBuffer>(this, handleType, handle, name, byteSize);
  }

  Ref<Semaphore> HIPEngine::newExternalSemaphore(ExternalSemaphoreTypeFlags fdType,
                                                  int fd)
  {
    return makeRef<HIPExternalSemaphore>(this, fdType, fd);
  }

  Ref<Semaphore> HIPEngine::newExternalSemaphore(ExternalSemaphoreTypeFlags handleType,
                                                  void* handle, const void* name)
  {
    return makeRef<HIPExternalSemaphore>(this, handleType, handle, name);
  }

  void HIPEngine::submitSignalSemaphores(Semaphore* const* semaphores,
                                          const uint64_t* values,
                                          int numSemaphores)
  {
    if (numSemaphores < 0)
      throw Exception(Error::InvalidArgument, "number of semaphores is negative");
    if (numSemaphores == 0)
      return;
    if (semaphores == nullptr)
      throw Exception(Error::InvalidArgument, "semaphores pointer is null");

    semaphoreHandles.resize(numSemaphores);
    semaphoreSignalParams.resize(numSemaphores);

    for (int i = 0; i < numSemaphores; ++i)
    {
      if (semaphores[i] == nullptr)
        throw Exception(Error::InvalidArgument, "semaphore is null");
      if (semaphores[i]->getDevice() != getDevice())
        throw Exception(Error::InvalidArgument, "semaphore was created on a different device");

      HIPExternalSemaphore* hipSemaphore = reinterpret_cast<HIPExternalSemaphore*>(semaphores[i]);
      ExternalSemaphoreTypeFlags type = hipSemaphore->getType();

      semaphoreHandles[i] = hipSemaphore->getHandle();

      semaphoreSignalParams[i] = {};
      if (values != nullptr)
      {
        if (type == ExternalSemaphoreTypeFlag::KeyedMutex ||
            type == ExternalSemaphoreTypeFlag::KeyedMutexKMT)
          semaphoreSignalParams[i].params.keyedMutex.key = values[i];
        else
          semaphoreSignalParams[i].params.fence.value = values[i];
      }
    }

    checkError(hipSignalExternalSemaphoresAsync(
      semaphoreHandles.data(),
      semaphoreSignalParams.data(),
      numSemaphores,
      stream));
  }

  void HIPEngine::submitWaitSemaphores(Semaphore* const* semaphores,
                                        const uint64_t* values,
                                        const uint32_t* timeoutsMs,
                                        int numSemaphores)
  {
    if (numSemaphores < 0)
      throw Exception(Error::InvalidArgument, "number of semaphores is negative");
    if (numSemaphores == 0)
      return;
    if (semaphores == nullptr)
      throw Exception(Error::InvalidArgument, "semaphores pointer is null");

    semaphoreHandles.resize(numSemaphores);
    semaphoreWaitParams.resize(numSemaphores);

    for (int i = 0; i < numSemaphores; ++i)
    {
      if (semaphores[i] == nullptr)
        throw Exception(Error::InvalidArgument, "semaphore is null");
      if (semaphores[i]->getDevice() != getDevice())
        throw Exception(Error::InvalidArgument, "semaphore was created on a different device");

      HIPExternalSemaphore* hipSemaphore = reinterpret_cast<HIPExternalSemaphore*>(semaphores[i]);
      ExternalSemaphoreTypeFlags type = hipSemaphore->getType();

      semaphoreHandles[i] = hipSemaphore->getHandle();

      semaphoreWaitParams[i] = {};
      if (type == ExternalSemaphoreTypeFlag::KeyedMutex ||
          type == ExternalSemaphoreTypeFlag::KeyedMutexKMT)
      {
        if (values != nullptr)
          semaphoreWaitParams[i].params.keyedMutex.key = values[i];
        if (timeoutsMs != nullptr)
          semaphoreWaitParams[i].params.keyedMutex.timeoutMs = timeoutsMs[i];
      }
      else
      {
        if (values != nullptr)
          semaphoreWaitParams[i].params.fence.value = values[i];
      }
    }

    checkError(hipWaitExternalSemaphoresAsync(
      semaphoreHandles.data(),
      semaphoreWaitParams.data(),
      numSemaphores,
      stream));
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