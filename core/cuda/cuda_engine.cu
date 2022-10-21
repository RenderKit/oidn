// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_engine.h"
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_pool.h"
#include "../gpu/gpu_upsample.h"
#include "../gpu/gpu_image_copy.h"
#include "cuda_conv.h"
#include "cuda_concat_conv.h"

namespace oidn {

  CUDAEngine::CUDAEngine(const Ref<CUDADevice>& device,
                         int deviceId,
                         cudaStream_t stream)
    : device(device.get()),
      deviceId(deviceId),
      stream(stream) {}

  void CUDAEngine::wait()
  {
    checkError(cudaStreamSynchronize(stream));
  }

  std::shared_ptr<Conv> CUDAEngine::newConv(const ConvDesc& desc)
  {
    return newCUDAConv(this, desc);
  }

  std::shared_ptr<ConcatConv> CUDAEngine::newConcatConv(const ConcatConvDesc& desc)
  {
    if (device->tensorLayout == TensorLayout::hwc)
      return std::make_shared<CUDAConcatConv>(this, desc);
    else
      return std::make_shared<CHWConcatConv>(this, desc);
  }

  std::shared_ptr<Pool> CUDAEngine::newPool(const PoolDesc& desc)
  {
    return std::make_shared<GPUPool<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<Upsample> CUDAEngine::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<GPUUpsample<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<Autoexposure> CUDAEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<GPUAutoexposure<CUDAEngine, 1024>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> CUDAEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<OutputProcess> CUDAEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<CUDAEngine, half, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<ImageCopy> CUDAEngine::newImageCopy()
  {
    return std::make_shared<GPUImageCopy<CUDAEngine>>(this);
  }

  void* CUDAEngine::malloc(size_t byteSize, Storage storage)
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

  void CUDAEngine::free(void* ptr, Storage storage)
  {
    if (storage == Storage::Host)
      checkError(cudaFreeHost(ptr));
    else
      checkError(cudaFree(ptr));
  }

  void CUDAEngine::memcpy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    checkError(cudaMemcpy(dstPtr, srcPtr, byteSize, cudaMemcpyDefault));
  }

  Storage CUDAEngine::getPointerStorage(const void* ptr)
  {
    cudaPointerAttributes attrib;
    if (cudaPointerGetAttributes(&attrib, ptr) != cudaSuccess)
      return Storage::Undefined;

    switch (attrib.type)
    {
    case cudaMemoryTypeHost:
      return Storage::Host;

    case cudaMemoryTypeDevice:
      return attrib.device == deviceId ? Storage::Device : Storage::Undefined;

    case cudaMemoryTypeManaged:
      return Storage::Managed;

    default:
      return Storage::Undefined;
    }
  }

  namespace
  {
    void CUDART_CB hostFuncCallback(cudaStream_t stream, cudaError_t status, void* fPtr)
    {
      std::unique_ptr<std::function<void()>> f(reinterpret_cast<std::function<void()>*>(fPtr));
      if (status == cudaSuccess)
        (*f)();
    }
  }

  void CUDAEngine::submitHostFunc(std::function<void()>&& f)
  {
    auto fPtr = new std::function<void()>(std::move(f));
    checkError(cudaStreamAddCallback(stream, hostFuncCallback, fPtr, 0));
  }
}