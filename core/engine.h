// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "kernel.h"
#include "heap.h"
#include "arena.h"
#include "buffer.h"
#include "tensor.h"
#include "image.h"
#include <functional>

OIDN_NAMESPACE_BEGIN

  struct ConvDesc;
  struct ConcatConvDesc;
  struct PoolDesc;
  struct UpsampleDesc;
  struct InputProcessDesc;
  struct OutputProcessDesc;

  enum class PostOp;
  class Conv;
  class ConcatConv;
  class Pool;
  class Upsample;
  class Autoexposure;
  class TransferFunction;
  class InputProcess;
  class OutputProcess;
  class ImageCopy;

  class ScratchArenaManager;

  // A device consists of one or more execution "engines"
  class Engine
  {
  public:
    Engine() = default;
    virtual ~Engine() = default;

    virtual Device* getDevice() const = 0;

    // Heap / arena
    virtual Ref<Heap> newHeap(size_t byteSize, Storage storage);
    Ref<Arena> newScratchArena(size_t byteSize, const std::string& name = "");
    void trimScratch();

    // Buffer
    virtual SizeAndAlignment getBufferByteSizeAndAlignment(size_t byteSize, Storage storage);
    virtual Ref<Buffer> newBuffer(size_t byteSize, Storage storage);
    virtual Ref<Buffer> newBuffer(void* ptr, size_t byteSize);
    virtual Ref<Buffer> newBuffer(const Ref<Arena>& arena, size_t byteSize, size_t byteOffset);

    virtual Ref<Buffer> newNativeBuffer(void* handle);

    virtual Ref<Buffer> newExternalBuffer(ExternalMemoryTypeFlag fdType,
                                          int fd, size_t byteSize);

    virtual Ref<Buffer> newExternalBuffer(ExternalMemoryTypeFlag handleType,
                                          void* handle, const void* name, size_t byteSize);

    // Tensor
    virtual bool isSupported(const TensorDesc& desc) const;
    virtual Ref<Tensor> newTensor(const TensorDesc& desc, Storage storage = Storage::Device);
    virtual Ref<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset = 0);
    std::shared_ptr<TensorMap> getCachedTensors(const void* key);

    // Ops
    virtual bool isConvSupported(PostOp postOp);
    virtual Ref<Conv> newConv(const ConvDesc& desc) = 0;
    virtual Ref<Pool> newPool(const PoolDesc& desc) = 0;
    virtual Ref<Upsample> newUpsample(const UpsampleDesc& desc) = 0;
    virtual Ref<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) = 0;
    virtual Ref<InputProcess> newInputProcess(const InputProcessDesc& desc) = 0;
    virtual Ref<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) = 0;
    virtual Ref<ImageCopy> newImageCopy() = 0;

    // Unified shared memory (USM)
    virtual void* usmAlloc(size_t byteSize, Storage storage);
    virtual void usmFree(void* ptr, Storage storage);
    virtual void usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize);
    virtual void submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize);

    // Runs a host task
    virtual void runHostTask(std::function<void()>&& f)
    {
      f();
    }

    // Enqueues a host function
    virtual void submitHostFunc(std::function<void()>&& f) = 0;

    // Issues all previously submitted commands (does not block)
    virtual void flush() {}

    // Waits for all previously submitted commands to complete (blocks)
    virtual void wait() = 0;

    virtual int getMaxWorkGroupSize() const;
    virtual int getSubgroupSize() const;

  protected:
    // Checks whether a kernel has a nested Local type for shared local memory
    template<typename T>
    struct HasLocal
    {
      template<typename U>
      static std::true_type test(typename U::Local*);

      template<typename U>
      static std::false_type test(...);

      static constexpr bool value = decltype(test<T>(nullptr))::value;
    };

  private:
    // Disable copying
    Engine(const Engine&) = delete;
    Engine& operator =(const Engine&) = delete;

    // Memory
    std::unique_ptr<ScratchArenaManager> scratchArenaManager;
    std::unordered_map<const void*, std::shared_ptr<TensorMap>> cachedTensors; // cached weights
  };

OIDN_NAMESPACE_END
