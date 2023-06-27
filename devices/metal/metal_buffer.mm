// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_buffer.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  MetalBuffer::MetalBuffer(const Ref<Engine>& engine,
                           size_t byteSize,
                           Storage storage)
    : engine(engine),
      byteSize(byteSize),
      storage((storage == Storage::Undefined) ? Storage::Host : storage),
      buffer(nullptr)
  {
    if (byteSize > 0)
      init();
  }

  MetalBuffer::~MetalBuffer()
  {
    free();
  }

  void MetalBuffer::init()
  {
    MTLDevice_t device = static_cast<MetalDevice*>(getDevice())->getMetalDevice();

    if (byteSize > [device maxBufferLength])
      throw std::bad_alloc();

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

    commandQueue = [device newCommandQueue];
  }

  void MetalBuffer::free()
  {
    if (buffer)
      [buffer release];
    buffer = nullptr;

    [commandQueue release];
    commandQueue = nullptr;
  }

  bool MetalBuffer::hasPtr() const
  {
    // Metal buffers with device/private storage do not have a pointer
    return storage != Storage::Device;
  }

  char* MetalBuffer::getPtr() const
  {
    return static_cast<char*>([buffer contents]);
  }

  void MetalBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    MTLDevice_t device = static_cast<MetalDevice*>(getDevice())->getMetalDevice();

    @autoreleasepool
    {
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceOptionCPUCacheModeDefault;
      id dstBuffer = [device newBufferWithBytesNoCopy: dstHostPtr
                                               length: byteSize
                                              options: options
                                          deallocator: nil];

      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

      [blitEncoder copyFromBuffer: buffer
                     sourceOffset: byteOffset
                         toBuffer: dstBuffer
                destinationOffset: 0
                             size: byteSize];
      [blitEncoder endEncoding];
      [commandBuffer commit];

      if (sync == SyncMode::Sync)
        [commandBuffer waitUntilCompleted];
    }
  }

  void MetalBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    MTLDevice_t device = static_cast<MetalDevice*>(getDevice())->getMetalDevice();

    @autoreleasepool
    {
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceOptionCPUCacheModeDefault;
      id srcBuffer = [device newBufferWithBytes: srcHostPtr
                                         length: byteSize
                                        options: options];

      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLBlitCommandEncoder> blitCommandEncoder = [commandBuffer blitCommandEncoder];

      [blitCommandEncoder copyFromBuffer: srcBuffer
                            sourceOffset: 0
                                toBuffer: buffer
                       destinationOffset: byteOffset
                                    size: byteSize];
      [blitCommandEncoder endEncoding];
      [commandBuffer commit];

      if (sync == SyncMode::Sync)
        [commandBuffer waitUntilCompleted];
    }
  }

  void MetalBuffer::realloc(size_t newByteSize)
  {
    free();

    byteSize = newByteSize;
    if (byteSize != 0)
      init();
  }

OIDN_NAMESPACE_END
