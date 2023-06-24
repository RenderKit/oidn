// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include "metal_buffer.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  MetalBuffer::MetalBuffer(const Ref<Engine>& engine,
                           size_t byteSize,
                           Storage storage)
    : engine(engine), byteSize(byteSize), storage(storage), buffer(nullptr)
  {
    if (byteSize > 0)
    {
      init();
    }
  }

  MetalBuffer::~MetalBuffer()
  {
    free();
  }

  void MetalBuffer::init()
  {
    MTLDevice_t device = static_cast<MetalDevice*>(getDevice())->getMetalDevice();
    MTLResourceOptions options = MTLResourceCPUCacheModeDefaultCache;
    
    if (byteSize > [device maxBufferLength])
    {
      throw std::bad_alloc();
    }
    
    switch (storage)
    {
    case Storage::Host:
        options |= MTLResourceStorageModeManaged;
        break;
    case Storage::Device:
        options |= MTLResourceStorageModeShared;
        break;
    case Storage::Managed:
        options |= MTLResourceStorageModeManaged;
        break;
    default:
        options |= MTLResourceStorageModeShared;
        break;
    }
    
    buffer = [device newBufferWithLength: byteSize
                                 options: options];
    
    commandQueue = [device newCommandQueue];
  }

  void MetalBuffer::free()
  {
    if (buffer)
    {
      [buffer release];
    }
    buffer = nullptr;
    
    [commandQueue release];
    commandQueue = nullptr;
  }

  char* MetalBuffer::getData()
  {
    return static_cast<char*>([buffer contents]);
  }

  const char* MetalBuffer::getData() const
  {
    return static_cast<const char*>([buffer contents]);
  }

  void* MetalBuffer::map(size_t byteOffset, size_t byteSize, Access access)
  {
    if (storage != Storage::Host)
      throw std::logic_error("map is not supported from on device memory");
    
    void* hostPtr = [buffer contents];
    
    mappedRegions.insert({hostPtr, {nullptr, byteOffset, byteSize, access}});
    
    return hostPtr;
  }

  void MetalBuffer::unmap(void* hostPtr)
  {
    auto region = mappedRegions.find(hostPtr);
    if (region == mappedRegions.end())
      throw Exception(Error::InvalidArgument, "invalid mapped region");
    if (region->second.access != Access::Read)
      [buffer didModifyRange: NSMakeRange(region->second.byteOffset, region->second.byteSize)];

    mappedRegions.erase(region);
  }

  void MetalBuffer::read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");
    
    MTLDevice_t device = static_cast<MetalDevice*>(getDevice())->getMetalDevice();
    @autoreleasepool {
      MTLResourceOptions options = MTLResourceOptionCPUCacheModeDefault | MTLResourceStorageModeShared;
      id dstBuffer = [device newBufferWithBytesNoCopy: dstHostPtr
                                               length: byteSize
                                              options: options
                                          deallocator: nil];
      
      id <MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLBlitCommandEncoder> blitEncoder = [commandBuffer blitCommandEncoder];

      [blitEncoder copyFromBuffer: buffer
                     sourceOffset: byteOffset
                         toBuffer: dstBuffer
                destinationOffset: 0
                             size: byteSize];
      [blitEncoder endEncoding];
      [commandBuffer commit];
      
      if (sync == SyncMode::Sync)
      {
        [commandBuffer waitUntilCompleted];
      }
    }
  }

  void MetalBuffer::write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync)
  {
    if (byteOffset + byteSize > this->byteSize)
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    MTLDevice_t device = static_cast<MetalDevice*>(getDevice())->getMetalDevice();
    @autoreleasepool {
      MTLResourceOptions options = MTLResourceOptionCPUCacheModeDefault | MTLResourceStorageModeShared;
      id srcBuffer = [device newBufferWithBytes: srcHostPtr
                                         length: byteSize
                                        options: options];
      
      id <MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id <MTLBlitCommandEncoder> blitCommandEncoder = [commandBuffer blitCommandEncoder];
      
      [blitCommandEncoder copyFromBuffer: srcBuffer
                            sourceOffset: 0
                                toBuffer: buffer
                       destinationOffset: byteOffset
                                    size: byteSize];
      [blitCommandEncoder endEncoding];
      [commandBuffer commit];
      
      if (sync == SyncMode::Sync)
      {
        [commandBuffer waitUntilCompleted];
      }
    }
  }

  void MetalBuffer::realloc(size_t newByteSize)
  {
    free();
    
    byteSize = newByteSize;
    if (byteSize != 0)
    {
      init();
    }
  }

OIDN_NAMESPACE_END
