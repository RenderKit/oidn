// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include "common.h"
#include "kernel.h"
#include "buffer.h"
#include "tensor_layout.h"

namespace oidn {

  struct TensorDesc;
  class Tensor;

  struct ImageDesc;
  class Image;
  
  class Filter;

  class ScratchBuffer;
  class ScratchBufferManager;

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

  class Device : public RefCount, public Verbose
  {
  public:
    Device();

    static void setError(Device* device, Error code, const std::string& message);
    static Error getError(Device* device, const char** outMessage);
    void setErrorFunction(ErrorFunction func, void* userPtr);

    void warning(const std::string& message);

    virtual int get1i(const std::string& name);
    virtual void set1i(const std::string& name, int value);

    bool isCommitted() const { return committed; }
    void checkCommitted();
    void commit();

    OIDN_INLINE Device* getDevice() { return this; }
    OIDN_INLINE std::mutex& getMutex() { return mutex; }

    // Native tensor layout
    DataType getTensorDataType() const { return tensorDataType; }
    TensorLayout getTensorLayout() const { return tensorLayout; }
    TensorLayout getWeightsLayout() const { return weightsLayout; }
    int getTensorBlockSize() const { return tensorBlockSize; }

    // Waits for all asynchronous operations to complete
    virtual void wait() {}

    virtual Ref<Buffer> newBuffer(size_t byteSize, Storage storage);
    virtual Ref<Buffer> newBuffer(void* ptr, size_t byteSize);

    Ref<ScratchBuffer> newScratchBuffer(size_t byteSize);

    Ref<Filter> newFilter(const std::string& type);

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
    virtual void* malloc(size_t byteSize, Storage storage);
    virtual void free(void* ptr, Storage storage);
    virtual void memcpy(void* dstPtr, const void* srcPtr, size_t byteSize);
    virtual Storage getPointerStorage(const void* ptr);

    // Runs a host task
    virtual void runHostTask(std::function<void()>&& f)
    {
      f();
    }

    // Enqueues a host function
    virtual void runHostFuncAsync(std::function<void()>&& f) = 0;
   
  protected:
    virtual void init() = 0;

    // Native tensor layout
    DataType tensorDataType = DataType::Float32;
    TensorLayout tensorLayout = TensorLayout::chw;
    TensorLayout weightsLayout = TensorLayout::oihw;
    int tensorBlockSize = 1;

    // State
    bool dirty = true;
    bool committed = false;

  private:
    // Thread-safety
    std::mutex mutex;

    // Error handling
    struct ErrorState
    {
      Error code = Error::None;
      std::string message;
    };

    static thread_local ErrorState globalError;
    ThreadLocal<ErrorState> error;
    ErrorFunction errorFunc = nullptr;
    void* errorUserPtr = nullptr;

    // Memory
    std::weak_ptr<ScratchBufferManager> scratchManagerWp;
  };

} // namespace oidn
