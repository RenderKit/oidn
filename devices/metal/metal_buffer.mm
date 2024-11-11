// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_buffer.h"
#include "core/arena.h"
#include "metal_heap.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  MetalBuffer::MetalBuffer(MetalEngine* engine,
                           size_t byteSize,
                           Storage storage)
    : engine(engine),
      buffer(nullptr),
      byteSize(byteSize),
      shared(false),
      storage((storage == Storage::Undefined) ? Storage::Host : storage)
  {
    // We disallow creating managed buffers because they would require manual synchronization
    if (storage == Storage::Managed)
      throw Exception(Error::InvalidArgument, "Metal managed storage mode is not supported");

    init();
  }

  MetalBuffer::MetalBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset)
    : Buffer(arena, byteOffset),
      engine(dynamic_cast<MetalEngine*>(arena->getEngine())),
      buffer(nullptr),
      byteSize(byteSize),
      shared(true),
      storage(arena->getHeap()->getStorage())
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

  MetalBuffer::MetalBuffer(MetalEngine* engine, void* data, size_t byteSize)
    : engine(engine),
      buffer(nullptr),
      byteSize(byteSize),
      shared(true),
      storage(Storage::Host)
  {
    if (data == nullptr)
      throw Exception(Error::InvalidArgument, "buffer pointer is null");
    if (byteSize == 0)
      return; // zero-sized Metal buffers cannot be created but we support them

    id<MTLDevice> device = engine->getMTLDevice();

    buffer = [device newBufferWithBytesNoCopy: data
                                       length: byteSize
                                      options: toMTLResourceOptions(storage)
                                  deallocator: nil];

    if (!buffer)
      throw Exception(Error::InvalidArgument, "failed to create buffer");
  }

  MetalBuffer::MetalBuffer(MetalEngine* engine, id<MTLBuffer> buffer)
    : engine(engine)
  {
    if (!buffer)
      throw Exception(Error::InvalidArgument, "Metal buffer is null");
    if (buffer.device != engine->getMTLDevice())
      throw Exception(Error::InvalidArgument, "Metal buffer belongs to a different device");
    if (buffer.hazardTrackingMode != MTLHazardTrackingModeTracked)
      throw Exception(Error::InvalidArgument, "Metal buffers without hazard tracking are not supported");

    switch (buffer.storageMode)
    {
    case MTLStorageModeShared:
      this->storage = Storage::Host;
      break;
    case MTLStorageModePrivate:
      this->storage = Storage::Device;
      break;
  #if TARGET_OS_OSX || TARGET_OS_MACCATALYST
    case MTLStorageModeManaged:
      this->storage = Storage::Managed; // we allow importing managed buffers
      break;
  #endif
    default:
      throw Exception(Error::InvalidArgument, "Metal buffer storage mode is not supported");
    }

    this->buffer   = [buffer retain];
    this->byteSize = buffer.length;
    this->shared   = true;
  }

  MetalBuffer::~MetalBuffer()
  {
    free();
  }

  void MetalBuffer::init()
  {
    if (byteSize == 0)
      return; // zero-sized Metal buffers cannot be created but we support them

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
    if (@available(macOS 13, iOS 16, tvOS 16, *))
      return reinterpret_cast<void*>([buffer gpuAddress]); // returns nullptr if buffer is nil
    else
      throw std::logic_error("getting the buffer pointer is not supported by the device");
  }

  void* MetalBuffer::getHostPtr() const
  {
    return storage != Storage::Device ? [buffer contents] : nullptr; // returns nullptr if buffer is nil
  }

  void MetalBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");
    if (dstHostPtr == nullptr && byteSize > 0)
      throw Exception(Error::InvalidArgument, "destination host pointer is null");

    @autoreleasepool
    {
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
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

      engine->sync(sync);
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
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceCPUCacheModeDefaultCache;
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

      engine->sync(sync);
    }
  }

OIDN_NAMESPACE_END
