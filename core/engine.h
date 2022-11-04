// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "kernel.h"
#include <functional>

namespace oidn {

  struct TensorDesc;
  class Tensor;

  struct ImageDesc;
  class Image;

  class TransferFunction;

  struct ConvDesc;
  struct ConcatConvDesc;
  struct PoolDesc;
  struct UpsampleDesc;
  struct InputProcessDesc;
  struct OutputProcessDesc;

  class Conv;
  class ConcatConv;
  class Pool;
  class Upsample;
  class Autoexposure;
  class InputProcess;
  class OutputProcess;
  class ImageCopy;

  class Buffer;
  class ScratchBuffer;
  class ScratchBufferManager;

  class Engine : public RefCount
  {
  public:
    virtual ~Engine() = default;

    virtual Device* getDevice() const = 0;

    virtual Ref<Buffer> newBuffer(size_t byteSize, Storage storage);
    virtual Ref<Buffer> newBuffer(void* ptr, size_t byteSize);
    Ref<ScratchBuffer> newScratchBuffer(size_t byteSize);

    virtual std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, Storage storage = Storage::Device);
    virtual std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, void* data);
    virtual std::shared_ptr<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset = 0);

    // Ops
    virtual std::shared_ptr<Conv> newConv(const ConvDesc& desc) = 0;
    virtual std::shared_ptr<ConcatConv> newConcatConv(const ConcatConvDesc& desc);
    virtual std::shared_ptr<Pool> newPool(const PoolDesc& desc) = 0;
    virtual std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) = 0;
    virtual std::shared_ptr<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) = 0;
    virtual std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) = 0;
    virtual std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) = 0;
    virtual std::shared_ptr<ImageCopy> newImageCopy() = 0;

    // Memory
    virtual void* malloc(size_t byteSize, Storage storage) = 0;
    virtual void free(void* ptr, Storage storage) = 0;
    virtual void memcpy(void* dstPtr, const void* srcPtr, size_t byteSize) = 0;
    virtual void submitMemcpy(void* dstPtr, const void* srcPtr, size_t byteSize) = 0;
    virtual Storage getPointerStorage(const void* ptr) = 0;

    // Runs a host task
    virtual void runHostTask(std::function<void()>&& f)
    {
      f();
    }

    // Enqueues a host function
    virtual void submitHostFunc(std::function<void()>&& f) = 0;

    // Waits for all asynchronous commands to complete (blocks)
    virtual void wait() = 0;

    virtual int getMaxWorkGroupSize() const { return 0; }

  private:
    // Memory
    std::weak_ptr<ScratchBufferManager> scratchManagerWp;
  };

} // namespace oidn
