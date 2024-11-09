// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/buffer.h"
#include "metal_engine.h"

OIDN_NAMESPACE_BEGIN

  class MetalBuffer : public Buffer
  {
  public:
    MetalBuffer(MetalEngine* engine, size_t byteSize, Storage storage);
    MetalBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset);
    MetalBuffer(MetalEngine* engine, void* data, size_t byteSize);
    MetalBuffer(MetalEngine* engine, id<MTLBuffer> buffer);
    ~MetalBuffer();

    Engine* getEngine() const override { return engine; }
    id<MTLBuffer> getMTLBuffer() const { return buffer; }
    void* getPtr() const override;
    void* getHostPtr() const override;
    size_t getByteSize() const override { return byteSize; }
    bool isShared() const override { return shared; }
    Storage getStorage() const override { return storage; }

    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr,
              SyncMode sync = SyncMode::Blocking) override;

    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr,
               SyncMode sync = SyncMode::Blocking) override;

  protected:
    void preRealloc() override;
    void postRealloc() override;

  private:
    void init();
    void free();

    MetalEngine* engine;
    id<MTLBuffer> buffer;
    size_t byteSize;
    bool shared;
    Storage storage;
  };

OIDN_NAMESPACE_END
