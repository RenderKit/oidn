// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_buffer.h"
#include "core/arena.h"
#include "metal_heap.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  MetalBuffer::MetalBuffer(const Ref<MetalEngine>& engine,
                           size_t byteSize,
                           Storage storage)
    : buffer(nullptr),
      byteSize(byteSize),
      storage((storage == Storage::Undefined) ? Storage::Host : storage),
      engine(engine)
  {
    init();
  }

  MetalBuffer::MetalBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset)
    : Buffer(arena, byteOffset),
      buffer(nullptr),
      byteSize(byteSize),
      storage(arena->getHeap()->getStorage()),
      engine(dynamic_cast<MetalEngine*>(arena->getEngine()))
  {
    if (!engine)
      throw Exception(Error::InvalidArgument, "buffer is incompatible with arena");
    const auto byteSizeAndAlignment = engine->getBufferByteSizeAndAlignment(byteSize, storage);
    if (byteOffset % byteSizeAndAlignment.alignment != 0)
      throw Exception(Error::InvalidArgument, "buffer offset is unaligned");
    if (byteOffset + byteSizeAndAlignment.size > arena->getByteSize())
      throw Exception(Error::InvalidArgument, "arena region is out of bounds");

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

    MTLResourceOptions options = toMTLResourceOptions(storage);

    if (arena)
    {
      MetalHeap* heap = static_cast<MetalHeap*>(arena->getHeap());

      buffer = [heap->heap newBufferWithLength: byteSize
                                       options: options
                                        offset: byteOffset];
    }
    else
    {
      buffer = [device newBufferWithLength: byteSize
                                   options: options];
    }

    if (!buffer)
      throw Exception(Error::OutOfMemory, "failed to create buffer");
  }

  void MetalBuffer::free()
  {
    if (buffer)
      [buffer release];
    buffer = nullptr;
  }

  void MetalBuffer::preRealloc()
  {
    Buffer::preRealloc();
    free();
  }

  void MetalBuffer::postRealloc()
  {
    init();
    Buffer::postRealloc();
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
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (dstHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "destination host pointer is null");

    @autoreleasepool
    {
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceOptionCPUCacheModeDefault;
      id<MTLBuffer> tempBuffer = [engine->getMTLDevice() newBufferWithLength: byteSize
                                                                     options: options];
      if (!tempBuffer)
        throw Exception(Error::OutOfMemory, "failed to create temporary buffer");

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

      if (sync == SyncMode::Sync)
        engine->wait();
      else
        engine->flush();
    }
  }

  void MetalBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (srcHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "source host pointer is null");

    @autoreleasepool
    {
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceOptionCPUCacheModeDefault;
      id<MTLBuffer> tempBuffer = [engine->getMTLDevice() newBufferWithBytes: srcHostPtr
                                                                     length: byteSize
                                                                    options: options];
      if (!tempBuffer)
        throw Exception(Error::OutOfMemory, "failed to create temporary buffer");

      id<MTLCommandBuffer> commandBuffer = engine->getMTLCommandBuffer();

      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];
      [blitEncoder copyFromBuffer: tempBuffer
                     sourceOffset: 0
                         toBuffer: buffer
                destinationOffset: byteOffset
                             size: byteSize];
      [blitEncoder endEncoding];

      [tempBuffer release];

      if (sync == SyncMode::Sync)
        engine->wait();
      else
        engine->flush();
    }
  }

OIDN_NAMESPACE_END
