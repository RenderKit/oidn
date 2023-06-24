// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "metal_device.h"

OIDN_NAMESPACE_BEGIN

  class MetalConcat;

  class MetalEngine : public Engine
  {
  public:
    explicit MetalEngine(const Ref<MetalDevice>& device);

    Device* getDevice() const override { return device; }

    // Buffer
    Ref<Buffer> newBuffer(size_t byteSize, Storage storage) override;
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize) override;

    // Tensor
    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, Storage storage = Storage::Device) override;
    std::shared_ptr<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset = 0) override;
    
    // Ops
    std::shared_ptr<Graph> newGraph(const std::shared_ptr<TensorMap>& constTensors, bool fastMath = false) override;
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;
    std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) override;
    std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    std::shared_ptr<ImageCopy> newImageCopy() override;

    // Memory
    void* usmAlloc(size_t byteSize, Storage storage) override;
    void usmFree(void* ptr, Storage storage) override;
    void usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;
    void submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;

    // Runs a parallel host task in the thread arena (if it exists)
    void runHostTask(std::function<void()>&& f) override;

    // Enqueues a host function
    void submitHostFunc(std::function<void()>&& f) override;

    void wait() override {}

  protected:
    MetalDevice* device;
  };

OIDN_NAMESPACE_END
