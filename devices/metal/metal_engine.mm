// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include "metal_engine.h"
#include "metal_buffer.h"
#include "metal_graph.h"
#include "metal_autoexposure.h"
#include "metal_input_process.h"
#include "metal_output_process.h"
#include "metal_image_copy.h"

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

  id<MTLComputePipelineState> MetalEngine::newMTLComputePipelineState(const std::string& functionName)
  {
    auto function = [library newFunctionWithName: @(functionName.c_str())];
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

  std::shared_ptr<Tensor> MetalEngine::newTensor(const TensorDesc& desc, Storage storage)
  {
    if (!isSupported(desc))
      throw std::invalid_argument("unsupported tensor descriptor");

    return std::make_shared<GenericTensor>(this, desc, storage);
  }

  std::shared_ptr<Tensor> MetalEngine::newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
  {
    if (!isSupported(desc))
      throw std::invalid_argument("unsupported tensor descriptor");
    if (buffer->getEngine() != this)
      throw std::invalid_argument("buffer was created by a different engine");
    if (byteOffset != 0)
      throw std::invalid_argument("unsupported tensor descriptor");

    return std::make_shared<GenericTensor>(buffer, desc, byteOffset);
  }

  std::shared_ptr<Graph> MetalEngine::newGraph(const std::shared_ptr<TensorMap>& constTensors, bool fastMath)
  {
    return std::make_shared<MetalGraph>(this, constTensors, fastMath);
  }

  std::shared_ptr<Conv> MetalEngine::newConv(const ConvDesc& desc)
  {
    throw std::logic_error("newConv is not supported");
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
    return std::make_shared<MetalAutoexposure>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> MetalEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<MetalInputProcess>(this, desc);
  }

  std::shared_ptr<OutputProcess> MetalEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<MetalOutputProcess>(this, desc);
  }

  std::shared_ptr<ImageCopy> MetalEngine::newImageCopy()
  {
    return std::make_shared<MetalImageCopy>(this);
  }

  void MetalEngine::submitHostFunc(std::function<void()>&& f)
  {
    f(); // no async execution on the Metal
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
