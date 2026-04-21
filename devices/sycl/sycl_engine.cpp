// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_engine.h"
#include "sycl_conv.h"
#include "sycl_external_buffer.h"
#include "sycl_external_semaphore.h"
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_image_copy.h"

OIDN_NAMESPACE_BEGIN

  SYCLEngine::SYCLEngine(SYCLDevice* device,
                         const sycl::queue& syclQueue)
    : device(device),
      syclQueue(syclQueue)
  {
    auto syclDevice = syclQueue.get_device();

    if (syclDevice.get_platform().get_backend() == sycl::backend::ext_oneapi_level_zero)
      zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclDevice);

    maxWorkGroupSize = syclDevice.get_info<sycl::info::device::max_work_group_size>();
  }

  Ref<Buffer> SYCLEngine::newExternalBuffer(ExternalMemoryTypeFlags fdType,
                                            int fd, size_t byteSize)
  {
    return makeRef<SYCLExternalBuffer>(this, fdType, fd, byteSize);
  }

  Ref<Buffer> SYCLEngine::newExternalBuffer(ExternalMemoryTypeFlags handleType,
                                            void* handle, const void* name, size_t byteSize)
  {
    return makeRef<SYCLExternalBuffer>(this, handleType, handle, name, byteSize);
  }

  Ref<Semaphore> SYCLEngine::newExternalSemaphore(ExternalSemaphoreTypeFlags fdType, int fd)
  {
    return makeRef<SYCLExternalSemaphore>(this, fdType, fd);
  }

  Ref<Semaphore> SYCLEngine::newExternalSemaphore(ExternalSemaphoreTypeFlags handleType,
                                                   void* handle, const void* name)
  {
    return makeRef<SYCLExternalSemaphore>(this, handleType, handle, name);
  }

  void SYCLEngine::submitSignalSemaphores(Semaphore* const* semaphores,
                                          const uint64_t* values,
                                          int numSemaphores)
  {
    if (numSemaphores < 0)
      throw Exception(Error::InvalidArgument, "number of semaphores is negative");
    if (numSemaphores == 0)
      return;
    if (semaphores == nullptr)
      throw Exception(Error::InvalidArgument, "semaphores pointer is null");

    auto depEvents = getDepEvents();
    std::vector<sycl::event> signalEvents;
    signalEvents.reserve(numSemaphores);

    for (int i = 0; i < numSemaphores; ++i)
    {
      if (semaphores[i] == nullptr)
        throw Exception(Error::InvalidArgument, "semaphore is null");
      if (semaphores[i]->getDevice() != getDevice())
        throw Exception(Error::InvalidArgument, "semaphore was created on a different device");

      SYCLExternalSemaphore* syclSem = reinterpret_cast<SYCLExternalSemaphore*>(semaphores[i]);
      sycl::event event;
      if (values != nullptr)
        event = syclQueue.ext_oneapi_signal_external_semaphore(syclSem->getHandle(), values[i], depEvents);
      else
        event = syclQueue.ext_oneapi_signal_external_semaphore(syclSem->getHandle(), depEvents);

      signalEvents.push_back(event);
    }

    if (numSemaphores == 1)
    {
      lastEvent = signalEvents[0];
    }
    else
    {
      lastEvent = syclQueue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(signalEvents);
        cgh.single_task([](){});
      });
    }
  }

  void SYCLEngine::submitWaitSemaphores(Semaphore* const* semaphores,
                                        const uint64_t* values,
                                        const uint32_t* /*timeoutsMs*/,
                                        int numSemaphores)
  {
    if (numSemaphores < 0)
      throw Exception(Error::InvalidArgument, "number of semaphores is negative");
    if (numSemaphores == 0)
      return;
    if (semaphores == nullptr)
      throw Exception(Error::InvalidArgument, "semaphores pointer is null");

    auto depEvents = getDepEvents();
    std::vector<sycl::event> waitEvents;
    waitEvents.reserve(numSemaphores);

    for (int i = 0; i < numSemaphores; ++i)
    {
      if (semaphores[i] == nullptr)
        throw Exception(Error::InvalidArgument, "semaphore is null");
      if (semaphores[i]->getDevice() != getDevice())
        throw Exception(Error::InvalidArgument, "semaphore was created on a different device");

      SYCLExternalSemaphore* syclSem = reinterpret_cast<SYCLExternalSemaphore*>(semaphores[i]);
      sycl::event event;
      if (values != nullptr)
        event = syclQueue.ext_oneapi_wait_external_semaphore(syclSem->getHandle(), values[i], depEvents);
      else
        event = syclQueue.ext_oneapi_wait_external_semaphore(syclSem->getHandle(), depEvents);

      waitEvents.push_back(event);
    }

    if (numSemaphores == 1)
    {
      lastEvent = waitEvents[0];
    }
    else
    {
      lastEvent = syclQueue.submit([&](sycl::handler& cgh) {
        cgh.depends_on(waitEvents);
        cgh.single_task([](){});
      });
    }
  }

  bool SYCLEngine::isConvSupported(PostOp postOp)
  {
    return postOp == PostOp::None ||
           postOp == PostOp::Pool ||
           postOp == PostOp::Upsample;
  }

  Ref<Conv> SYCLEngine::newConv(const ConvDesc& desc)
  {
    switch (device->getArch())
    {
    case SYCLArch::Xe_NoDPAS:
    case SYCLArch::XeLP_NoDPAS:
    case SYCLArch::XeLPG_NoDPAS:
    case SYCLArch::XeHPC_NoDPAS:
      return xelp::newSYCLConv(this, desc);

    case SYCLArch::Xe:
    case SYCLArch::XeLPGplus:
    case SYCLArch::XeHPG:
      return xehpg::newSYCLConv(this, desc);

  #if defined(__linux__)
    case SYCLArch::XeHPC:
      return xehpc::newSYCLConv(this, desc);
  #endif

    case SYCLArch::Xe2:
    case SYCLArch::Xe2LPG:
    case SYCLArch::Xe2HPG:
    case SYCLArch::Xe3:
    case SYCLArch::Xe3LPG:
    case SYCLArch::Xe3pXPC:
      return xe2::newSYCLConv(this, desc);

    default:
      throw std::logic_error("unsupported architecture");
    }
  }

  Ref<Pool> SYCLEngine::newPool(const PoolDesc& desc)
  {
    throw std::logic_error("operation is not implemented");
  }

  Ref<Upsample> SYCLEngine::newUpsample(const UpsampleDesc& desc)
  {
    throw std::logic_error("operation is not implemented");
  }

  Ref<Autoexposure> SYCLEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    if (maxWorkGroupSize >= 1024)
      return makeRef<GPUAutoexposure<SYCLEngine, 1024>>(this, srcDesc);
    else if (maxWorkGroupSize >= 512)
      return makeRef<GPUAutoexposure<SYCLEngine, 512>>(this, srcDesc);
    else
      return makeRef<GPUAutoexposure<SYCLEngine, 256>>(this, srcDesc);
  }

  Ref<InputProcess> SYCLEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return makeRef<GPUInputProcess<SYCLEngine, half, TensorLayout::Chw16c, 16>>(this, desc);
  }

  Ref<OutputProcess> SYCLEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return makeRef<GPUOutputProcess<SYCLEngine, half, TensorLayout::Chw16c>>(this, desc);
  }

  Ref<ImageCopy> SYCLEngine::newImageCopy()
  {
    return makeRef<GPUImageCopy<SYCLEngine>>(this);
  }

  void* SYCLEngine::usmAlloc(size_t byteSize, Storage storage)
  {
    if (byteSize == 0)
      return nullptr;

    void* ptr = nullptr;

    switch (storage)
    {
    case Storage::Host:
      ptr = sycl::aligned_alloc_host(memoryAlignment,
                                     byteSize,
                                     syclQueue.get_context());
      break;

    case Storage::Device:
      ptr = sycl::aligned_alloc_device(memoryAlignment,
                                       byteSize,
                                       syclQueue.get_device(),
                                       syclQueue.get_context());
      break;

    case Storage::Managed:
      ptr = sycl::aligned_alloc_shared(memoryAlignment,
                                       byteSize,
                                       syclQueue.get_device(),
                                       syclQueue.get_context());
      break;

    default:
      throw Exception(Error::InvalidArgument, "invalid storage mode");
    }

    if (ptr == nullptr && byteSize > 0)
      throw std::bad_alloc();

    return ptr;
  }

  void SYCLEngine::usmFree(void* ptr, Storage storage)
  {
    if (ptr != nullptr)
      sycl::free(ptr, syclQueue.get_context());
  }

  void SYCLEngine::usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    submitUSMCopy(dstPtr, srcPtr, byteSize);
    wait();
  }

  void SYCLEngine::submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    lastEvent = syclQueue.memcpy(dstPtr, srcPtr, byteSize, getDepEvents());
  }

  void SYCLEngine::submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct)
  {
    lastEvent = syclQueue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(getDepEvents()),
      cgh.host_task(f);
    });
  }

  void SYCLEngine::submitBarrier()
  {
    lastEvent = syclQueue.submit([&](sycl::handler& cgh) {
      cgh.depends_on(getDepEvents()),
      //cgh.ext_oneapi_barrier(); // FIXME: hangs, workaround: SYCL_PI_LEVEL_ZERO_USE_MULTIPLE_COMMANDLIST_BARRIERS=0
      cgh.single_task([](){});    // FIXME: should switch to ext_oneapi_barrier when it gets fixed
    });
  }

  void SYCLEngine::wait()
  {
    if (lastEvent)
    {
      lastEvent.value().wait_and_throw();
      lastEvent.reset();
    }
  }

OIDN_NAMESPACE_END
