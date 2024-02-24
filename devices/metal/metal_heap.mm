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
    free();
    byteSize = newByteSize;
    init();
    postRealloc();
  }

OIDN_NAMESPACE_END