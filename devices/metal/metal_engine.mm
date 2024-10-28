// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_engine.h"
#include "metal_heap.h"
#include "metal_buffer.h"
#include "metal_conv.h"
#include "devices/metal/metal_kernels.h" // generated
#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_image_copy.h"

OIDN_NAMESPACE_BEGIN

  MetalEngine::MetalEngine(MetalDevice* device)
    : device(device)
  {
    if (device->userCommandQueue)
      commandQueue = [device->userCommandQueue retain];
    else
    {
      commandQueue = [device->getMTLDevice() newCommandQueue];
      if (!commandQueue)
        throw std::runtime_error("could not create Metal command queue");
    }

    dispatch_data_t libraryData = dispatch_data_create(blobs::metal_kernels, sizeof(blobs::metal_kernels),
                                                       nil, DISPATCH_DATA_DESTRUCTOR_DEFAULT);

    NSError* error = nullptr;
    library = [device->getMTLDevice() newLibraryWithData: libraryData
                                                   error: &error];

    dispatch_release(libraryData);

    if (!library)
      throw std::runtime_error("could not create Metal library");
  }

  MetalEngine::~MetalEngine()
  {
    [library release];
    if (commandBuffer)
      [commandBuffer release];
    [commandQueue release];
  }

  id<MTLCommandBuffer> MetalEngine::getMTLCommandBuffer()
  {
    return getMPSCommandBuffer();
  }

  MPSCommandBuffer* MetalEngine::getMPSCommandBuffer()
  {
    if (!commandBuffer)
      commandBuffer = [[MPSCommandBuffer commandBufferFromCommandQueue: commandQueue] retain];
    return commandBuffer;
  }

  Ref<Heap> MetalEngine::newHeap(size_t byteSize, Storage storage)
  {
    return makeRef<MetalHeap>(this, byteSize, storage);
  }

  SizeAndAlignment MetalEngine::getBufferByteSizeAndAlignment(size_t byteSize, Storage storage)
  {
    MTLSizeAndAlign sizeAndAlign =
      [device->getMTLDevice() heapBufferSizeAndAlignWithLength: byteSize
                                                       options: toMTLResourceOptions(storage)];

    return {sizeAndAlign.size, sizeAndAlign.align};
  }

  Ref<Buffer> MetalEngine::newBuffer(size_t byteSize, Storage storage)
  {
    return makeRef<MetalBuffer>(this, byteSize, storage);
  }

  Ref<Buffer> MetalEngine::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<MetalBuffer>(this, ptr, byteSize);
  }

  Ref<Buffer> MetalEngine::newBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset)
  {
    return makeRef<MetalBuffer>(arena, byteSize, byteOffset);
  }

  Ref<Buffer> MetalEngine::newNativeBuffer(void* handle)
  {
    return makeRef<MetalBuffer>(this, (__bridge id<MTLBuffer>)handle);
  }

  Ref<Tensor> MetalEngine::newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc,
                                     size_t byteOffset)
  {
    // MPS requires the tensor to be in its own buffer so we suballocate a new buffer from the
    // provided one, if possible
    if (buffer->getArena())
      return makeRef<DeviceTensor>(buffer->newBuffer(desc.getByteSize(), byteOffset), desc);
    else
      return makeRef<DeviceTensor>(buffer, desc, byteOffset);
  }

  bool MetalEngine::isConvSupported(PostOp postOp)
  {
    return postOp == PostOp::None ||
           postOp == PostOp::Pool ||
           postOp == PostOp::Upsample;
  }

  Ref<Conv> MetalEngine::newConv(const ConvDesc& desc)
  {
    return makeRef<MetalConv>(this, desc);
  }

  Ref<Pool> MetalEngine::newPool(const PoolDesc& desc)
  {
    throw std::logic_error("operation is not implemented");
  }

  Ref<Upsample> MetalEngine::newUpsample(const UpsampleDesc& desc)
  {
    throw std::logic_error("operation is not implemented");
  }

  Ref<Autoexposure> MetalEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return makeRef<GPUAutoexposure<MetalEngine, 1024>>(this, srcDesc);
  }

  Ref<InputProcess> MetalEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return makeRef<GPUInputProcess<MetalEngine, half, TensorLayout::hwc, 1>>(this, desc);
  }

  Ref<OutputProcess> MetalEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return makeRef<GPUOutputProcess<MetalEngine, half, TensorLayout::hwc>>(this, desc);
  }

  Ref<ImageCopy> MetalEngine::newImageCopy()
  {
    return makeRef<GPUImageCopy<MetalEngine>>(this);
  }

  void MetalEngine::submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct)
  {
    auto fPtr = new std::function<void()>(std::move(f));

    auto commandBuffer = getMTLCommandBuffer();
    [commandBuffer addCompletedHandler: ^(id<MTLCommandBuffer> commandBuffer)
    {
      std::unique_ptr<std::function<void()>> fSmartPtr(fPtr);
      (*fSmartPtr)();
    }];

    flush();
  }

  Ref<MetalPipeline> MetalEngine::newPipeline(const std::string& kernelName)
  {
    auto function = [[library newFunctionWithName: @(kernelName.c_str())] autorelease];
    if (!function)
      throw std::runtime_error("could not create Metal library function");

    NSError* error = nullptr;
    auto pipelineState = [device->getMTLDevice() newComputePipelineStateWithFunction: function
                                                                               error: &error];

    if (!pipelineState)
      throw std::runtime_error("could not create Metal compute pipeline state");

    return makeRef<MetalPipeline>(pipelineState);
  }

  void MetalEngine::flush()
  {
    if (commandBuffer)
      [commandBuffer commitAndContinue];
  }

  void MetalEngine::wait()
  {
    if (commandBuffer)
    {
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      [commandBuffer release];
      commandBuffer = nullptr;
    }
  }

OIDN_NAMESPACE_END
