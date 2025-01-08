// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_engine.h"
#include "sycl_conv.h"
#include "sycl_external_buffer.h"
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

  Ref<Buffer> SYCLEngine::newExternalBuffer(ExternalMemoryTypeFlag fdType,
                                            int fd, size_t byteSize)
  {
    return makeRef<SYCLExternalBuffer>(this, fdType, fd, byteSize);
  }

  Ref<Buffer> SYCLEngine::newExternalBuffer(ExternalMemoryTypeFlag handleType,
                                            void* handle, const void* name, size_t byteSize)
  {
    return makeRef<SYCLExternalBuffer>(this, handleType, handle, name, byteSize);
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
    case SYCLArch::XeLP:
    case SYCLArch::XeLPG:
    case SYCLArch::XeHPC_NoDPAS:
      return xelp::newSYCLConv(this, desc);
    case SYCLArch::XeLPGplus:
    case SYCLArch::XeHPG:
      return xehpg::newSYCLConv(this, desc);
  #if defined(__linux__)
    case SYCLArch::XeHPC:
      return xehpc::newSYCLConv(this, desc);
  #endif
    case SYCLArch::Xe2LPG:
    case SYCLArch::Xe2HPG:
    case SYCLArch::Xe3LPG:
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
