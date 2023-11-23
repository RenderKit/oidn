// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/buffer.h"
#include "metal_common.h"

OIDN_NAMESPACE_BEGIN

  class MetalEngine;

  class MetalBuffer : public Buffer
  {
  public:
    MetalBuffer(const Ref<MetalEngine>& engine, size_t byteSize, Storage storage);
    MetalBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset);
    MetalBuffer(const Ref<MetalEngine>& engine, id<MTLBuffer> buffer);
    ~MetalBuffer();

    Engine* getEngine() const override { return (Engine*)engine.get(); }
    id<MTLBuffer> getMTLBuffer() const { return buffer; }
    void* getPtr() const override;
    void* getHostPtr() const override;
    size_t getByteSize() const override { return byteSize; }
    Storage getStorage() const override { return storage; }

    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr, SyncMode sync = SyncMode::Sync) override;
    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr, SyncMode sync = SyncMode::Sync) override;

  protected:
    void preRealloc() override;
    void postRealloc() override;

  private:
    void init();
    void free();

    id<MTLBuffer> buffer;
    size_t byteSize;
    Storage storage;
    Ref<MetalEngine> engine;
  };

OIDN_NAMESPACE_END
