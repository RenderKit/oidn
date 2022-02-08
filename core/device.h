// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"
#include "buffer.h"
#include "tensor_layout.h"

namespace oidn {

  struct TensorDesc;
  class Tensor;

  class Image;
  class Filter;

  class ScratchBuffer;
  class ScratchBufferManager;

  class TransferFunction;

  struct ConvDesc;
  struct PoolDesc;
  struct UpsampleDesc;
  struct InputProcessDesc;
  struct OutputProcessDesc;

  class Conv;
  class Pool;
  class Upsample;
  class InputProcess;
  class OutputProcess;

  struct ConcatConvDesc;
  class ConcatConv;

  class Device : public RefCount, public Verbose
  {
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

  protected:
    // Tasking
    std::shared_ptr<tbb::task_arena> arena;

    // Native tensor layout
    DataType tensorDataType = DataType::Float32;
    TensorLayout tensorLayout = TensorLayout::chw;
    TensorLayout weightsLayout = TensorLayout::oihw;
    int tensorBlockSize = 1;

    // Parameters
    int numThreads = 0; // autodetect by default
    bool setAffinity = true;

    bool dirty = true;
    bool committed = false;

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

    virtual void wait() {}

    virtual Ref<Buffer> newBuffer(size_t byteSize, MemoryKind kind) = 0;
    virtual Ref<Buffer> newBuffer(void* ptr, size_t byteSize) = 0;

    Ref<ScratchBuffer> newScratchBuffer(size_t byteSize);

    Ref<Filter> newFilter(const std::string& type);

    virtual std::shared_ptr<Tensor> newTensor(const TensorDesc& desc);
    virtual std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, void* data);
    virtual std::shared_ptr<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset);

    // Ops
    virtual std::shared_ptr<Conv> newConv(const ConvDesc& desc) = 0;
    virtual std::shared_ptr<ConcatConv> newConcatConv(const ConcatConvDesc& desc) { return nullptr; }
    virtual std::shared_ptr<Pool> newPool(const PoolDesc& desc) = 0;
    virtual std::shared_ptr<Upsample> newUpsample(const UpsampleDesc& desc) = 0;
    virtual std::shared_ptr<InputProcess> newInputProcess(const InputProcessDesc& desc) = 0;
    virtual std::shared_ptr<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) = 0;

    // Kernels
    virtual void imageCopy(const Image& src, const Image& dst) = 0;

    // Runs task in the arena (if it exists)
    template<typename F>
    void runTask(const F& f)
    {
      if (arena)
        arena->execute(f);
      else
        f();
    }
   
  protected:
    virtual void init() = 0;
    virtual void printInfo() = 0;
  };

} // namespace oidn
