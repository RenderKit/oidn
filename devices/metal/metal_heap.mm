// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_heap.h"

OIDN_NAMESPACE_BEGIN

  MetalHeap::MetalHeap(MetalEngine* engine, size_t byteSize, Storage storage)
    : engine(engine),
      heap(nullptr),
      byteSize(byteSize),
      storage((storage == Storage::Undefined) ? Storage::Device : storage)
  {
    init();
  }

  MetalHeap::~MetalHeap()
  {
    free();
  }

  void MetalHeap::init()
  {
    if (byteSize == 0)
      return;

    MTLHeapDescriptor* desc = [MTLHeapDescriptor new];
    desc.type = MTLHeapTypePlacement;
    desc.resourceOptions = toMTLResourceOptions(storage) | MTLResourceHazardTrackingModeTracked;
    desc.size = engine->getBufferByteSizeAndAlignment(byteSize, storage).size;

    heap = [engine->getMTLDevice() newHeapWithDescriptor: desc];
    [desc release];

    if (!heap)
      throw Exception(Error::OutOfMemory, "failed to create heap");
  }

  void MetalHeap::free()
  {
    if (heap)
      [heap release];
    heap = nullptr;
  }

  void MetalHeap::realloc(size_t newByteSize)
  {
    if (newByteSize == byteSize)
      return;

    preRealloc();

    // The old heap is released before creating the new one to avoid having to hold both at the
    // same time, which would increase the peak memory usage
    free();
    byteSize = newByteSize;

    try
    {
      init();
    }
    catch (...)
    {
      // No attempt is made to create a heap of the old size again: running out of memory is
      // usually not recoverable, and it could fail as well. The heap is simply left empty, but
      // the buffers attached to it must be still updated, otherwise they would keep pointing to
      // the memory which has just been released.
      byteSize = 0;
      postRealloc();
      throw;
    }

    postRealloc();
  }

OIDN_NAMESPACE_END