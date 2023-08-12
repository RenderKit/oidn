// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_buffer.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  MetalBuffer::MetalBuffer(const Ref<MetalEngine>& engine,
                           size_t byteSize,
                           Storage storage)
    : engine(engine),
      byteSize(byteSize),
      storage((storage == Storage::Undefined) ? Storage::Host : storage),
      buffer(nullptr)
  {
    init();
  }

  MetalBuffer::~MetalBuffer()
  {
    free();
  }

  void MetalBuffer::init()
  {
    if (byteSize == 0)
      return;

    id<MTLDevice> device = engine->getMTLDevice();

    if (byteSize > [device maxBufferLength])
      throw Exception(Error::OutOfMemory, "buffer size exceeds device limit");

    MTLResourceOptions options;
    switch (storage)
    {
    case Storage::Host:
      options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
      break;
    case Storage::Device:
      options = MTLResourceStorageModePrivate;
      break;
    default:
      throw Exception(Error::InvalidArgument, "invalid storage mode");
    }

    buffer = [device newBufferWithLength: byteSize
                                 options: options];
    if (!buffer)
      throw Exception(Error::OutOfMemory, "failed to allocate buffer");
  }

  void MetalBuffer::free()
  {
    if (buffer)
      [buffer release];
    buffer = nullptr;
  }

  void* MetalBuffer::getPtr() const
  {
    return reinterpret_cast<void*>([buffer gpuAddress]);
  }

  void* MetalBuffer::getHostPtr() const
  {
    return storage != Storage::Device ? [buffer contents] : nullptr;
  }

  void MetalBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    @autoreleasepool
    {
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceOptionCPUCacheModeDefault;
      id<MTLBuffer> tempBuffer = [engine->getMTLDevice() newBufferWithLength: byteSize
                                                                     options: options];
      if (!tempBuffer)
        throw Exception(Error::OutOfMemory, "failed to allocate temporary buffer");

      id<MTLCommandBuffer> commandBuffer = engine->getMTLCommandBuffer();

      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      [blitEncoder copyFromBuffer: buffer
                     sourceOffset: byteOffset
                         toBuffer: tempBuffer
                destinationOffset: 0
                             size: byteSize];
      [blitEncoder endEncoding];

      [commandBuffer addCompletedHandler: ^(id<MTLCommandBuffer> commandBuffer)
      {
        memcpy(dstHostPtr, [tempBuffer contents], byteSize);
        [tempBuffer release];
      }];

      [commandBuffer commit];

      if (sync == SyncMode::Sync)
        engine->wait();
    }
  }

  void MetalBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    @autoreleasepool
    {
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceOptionCPUCacheModeDefault;
      id<MTLBuffer> tempBuffer = [engine->getMTLDevice() newBufferWithBytes: srcHostPtr
                                                                     length: byteSize
                                                                    options: options];
      if (!tempBuffer)
        throw Exception(Error::OutOfMemory, "failed to allocate temporary buffer");

      id<MTLCommandBuffer> commandBuffer = engine->getMTLCommandBuffer();

      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      [blitEncoder copyFromBuffer: tempBuffer
                     sourceOffset: 0
                         toBuffer: buffer
                destinationOffset: byteOffset
                             size: byteSize];
      [blitEncoder endEncoding];

      [commandBuffer commit];
      [tempBuffer release];

      if (sync == SyncMode::Sync)
        engine->wait();
    }
  }

  void MetalBuffer::realloc(size_t newByteSize)
  {
    free();

    byteSize = newByteSize;
    init();
  }

OIDN_NAMESPACE_END
