// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "subdevice.h"
#include "kernel.h"
#include "heap.h"
#include "buffer.h"
#include "image.h"
#include "progress.h"

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
  struct TransferFunction;
  class InputProcess;
  class OutputProcess;
  class ImageCopy;

  // Execution engine of a subdevice
  class Engine
  {
  public:
    Engine() = default;
    virtual ~Engine() = default;

    virtual Device* getDevice() const = 0;
    Subdevice* getSubdevice() const { return subdevice; }
    void setSubdevice(Subdevice* subdevice); // set once by Subdevice

    // Heap
    virtual Ref<Heap> newHeap(size_t byteSize, Storage storage);

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

    // Enqueues a host function
    virtual void submitHostFunc(std::function<void()>&& f,
                                const Ref<CancellationToken>& ct = nullptr) = 0;

    // Issues all previously submitted commands (does not block)
    virtual void flush() {}

    // Waits for all previously submitted commands to complete (blocks)
    virtual void wait() = 0;

    // Calls wait() or flush() depending on the sync mode
    void sync(SyncMode syncMode)
    {
      if (syncMode == SyncMode::Blocking)
        wait();
      else
        flush();
    }

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

    Subdevice* subdevice = nullptr; // each engine belongs to a subdevice
  };

OIDN_NAMESPACE_END
