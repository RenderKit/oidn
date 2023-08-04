// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_engine.h"
#include "metal_buffer.h"
#include "metal_graph.h"
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_image_copy.h"

OIDN_NAMESPACE_BEGIN

  MetalEngine::MetalEngine(const Ref<MetalDevice>& device)
    : device(device.get()),
      commandQueue([device->getMTLDevice() newCommandQueue]),
      lastCommandBuffer(nil),
      library([device->getMTLDevice() newDefaultLibrary])
  {
    if (!library)
      throw std::runtime_error("could not create default Metal library");
  }

  MetalEngine::~MetalEngine()
  {
    [library release];
    if (lastCommandBuffer)
      [lastCommandBuffer release];
    [commandQueue release];
  }

  id<MTLComputePipelineState> MetalEngine::newMTLComputePipelineState(const std::string& kernelName)
  {
    auto function = [library newFunctionWithName: @(kernelName.c_str())];
    if (!function)
      throw std::runtime_error("could not create Metal library function");

    NSError* error = nil;
    auto pipeline = [device->getMTLDevice() newComputePipelineStateWithFunction: function
                                            error: &error];

    if (!pipeline)
      throw std::runtime_error("could not create Metal compute pipeline state");

    return pipeline;
  }

  id<MTLCommandBuffer> MetalEngine::getMTLCommandBuffer()
  {
    if (lastCommandBuffer)
      [lastCommandBuffer release];

    lastCommandBuffer = [commandQueue commandBuffer].retain;
    return lastCommandBuffer;
  }

  MPSCommandBuffer* MetalEngine::getMPSCommandBuffer()
  {
    if (lastCommandBuffer)
      [lastCommandBuffer release];

    MPSCommandBuffer* mpsCommandBuffer =
      [MPSCommandBuffer commandBufferFromCommandQueue: commandQueue].retain;
    lastCommandBuffer = mpsCommandBuffer;
    return mpsCommandBuffer;
  }

  Ref<Buffer> MetalEngine::newBuffer(size_t byteSize, Storage storage)
  {
    return makeRef<MetalBuffer>(this, byteSize, storage);
  }

  Ref<Buffer> MetalEngine::newBuffer(void* ptr, size_t byteSize)
  {
    throw Exception(Error::InvalidOperation, "creating shared buffers is not supported by the device");
  }

  void MetalEngine::runHostTask(std::function<void()>&& f)
  {
    @autoreleasepool
    {
      f();
    }
  }

  std::shared_ptr<Graph> MetalEngine::newGraph(const std::shared_ptr<TensorMap>& constTensors, bool fastMath)
  {
    return std::make_shared<MetalGraph>(this, constTensors);
  }

  std::shared_ptr<Conv> MetalEngine::newConv(const ConvDesc& desc)
  {
    return std::make_shared<MetalConv>(desc);
  }

  std::shared_ptr<Pool> MetalEngine::newPool(const PoolDesc& desc)
  {
    throw std::logic_error("newPool is not supported");
  }

  std::shared_ptr<Upsample> MetalEngine::newUpsample(const UpsampleDesc& desc)
  {
    throw std::logic_error("newUpsample is not supported");
  }

  std::shared_ptr<Autoexposure> MetalEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<GPUAutoexposure<MetalEngine, 1024>>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> MetalEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<GPUInputProcess<MetalEngine, float, TensorLayout::hwc, 1>>(this, desc);
  }

  std::shared_ptr<OutputProcess> MetalEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<GPUOutputProcess<MetalEngine, float, TensorLayout::hwc>>(this, desc);
  }

  std::shared_ptr<ImageCopy> MetalEngine::newImageCopy()
  {
    return std::make_shared<GPUImageCopy<MetalEngine>>(this);
  }

  void MetalEngine::submitHostFunc(std::function<void()>&& f)
  {
    auto fPtr = new std::function<void()>(std::move(f));
    auto commandBuffer = getMTLCommandBuffer();

    [commandBuffer addCompletedHandler: ^(id<MTLCommandBuffer> commandBuffer)
    {
      std::unique_ptr<std::function<void()>> fSmartPtr(fPtr);
      (*fSmartPtr)();
    }];

    [commandBuffer commit];
  }

  void MetalEngine::wait()
  {
    if (lastCommandBuffer)
    {
      [lastCommandBuffer waitUntilCompleted];
      [lastCommandBuffer release];
      lastCommandBuffer = nil;
    }
  }

OIDN_NAMESPACE_END
