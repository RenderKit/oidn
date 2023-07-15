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
  }

  void MetalBuffer::free()
  {
    if (buffer)
      [buffer release];
    buffer = nullptr;
  }

  char* MetalBuffer::getPtr() const
  {
    return reinterpret_cast<char*>([buffer gpuAddress]);
  }

  char* MetalBuffer::getHostPtr() const
  {
    return storage != Storage::Device ? static_cast<char*>([buffer contents]) : nullptr;
  }

  void MetalBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    id<MTLDevice> device = static_cast<MetalDevice*>(getDevice())->getMTLDevice();

    @autoreleasepool
    {
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceOptionCPUCacheModeDefault;
      id dstBuffer = [device newBufferWithBytesNoCopy: dstHostPtr
                                               length: byteSize
                                              options: options
                                          deallocator: nil];

      id<MTLCommandBuffer> commandBuffer = engine->getMTLCommandBuffer();
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

      [blitEncoder copyFromBuffer: buffer
                     sourceOffset: byteOffset
                         toBuffer: dstBuffer
                destinationOffset: 0
                             size: byteSize];
      [blitEncoder endEncoding];
      [commandBuffer commit];

      if (sync == SyncMode::Sync)
        engine->wait();
    }
  }

  void MetalBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    id<MTLDevice> device = static_cast<MetalDevice*>(getDevice())->getMTLDevice();

    @autoreleasepool
    {
      const MTLResourceOptions options = MTLResourceStorageModeShared | MTLResourceOptionCPUCacheModeDefault;
      id srcBuffer = [device newBufferWithBytes: srcHostPtr
                                         length: byteSize
                                        options: options];

      id<MTLCommandBuffer> commandBuffer = engine->getMTLCommandBuffer();
      id<MTLBlitCommandEncoder> blitCommandEncoder = [commandBuffer blitCommandEncoder];

      [blitCommandEncoder copyFromBuffer: srcBuffer
                            sourceOffset: 0
                                toBuffer: buffer
                       destinationOffset: byteOffset
                                    size: byteSize];
      [blitCommandEncoder endEncoding];
      [commandBuffer commit];

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
