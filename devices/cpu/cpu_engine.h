// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/engine.h"
#include "cpu_device.h"
#include <queue>
#include <thread>
#include <condition_variable>

OIDN_NAMESPACE_BEGIN

  class CPUEngine : public Engine
  {
    friend class CPUDevice;

  public:
    CPUEngine(CPUDevice* device, int numThreads);
    ~CPUEngine();

    Device* getDevice() const override { return device; }
    int getNumThreads() const { return device->numThreads; }

    // Ops
  #if !defined(OIDN_DNNL) && !defined(OIDN_BNNS)
    Ref<Conv> newConv(const ConvDesc& desc) override;
  #endif
    Ref<Pool> newPool(const PoolDesc& desc) override;
    Ref<Upsample> newUpsample(const UpsampleDesc& desc) override;
    Ref<Autoexposure> newAutoexposure(const ImageDesc& srcDesc) override;
    Ref<InputProcess> newInputProcess(const InputProcessDesc& desc) override;
    Ref<OutputProcess> newOutputProcess(const OutputProcessDesc& desc) override;
    Ref<ImageCopy> newImageCopy() override;

    // Unified shared memory (USM)
    void* usmAlloc(size_t byteSize, Storage storage) override;
    void usmFree(void* ptr, Storage storage) override;
    void usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;
    void submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize) override;

    // Enqueues a function
    void submitFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct = nullptr);

    // Enqueues a host function
    void submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct) override;

    void wait() override;

  protected:
    struct Task
    {
      std::function<void()> func;
      Ref<CancellationToken> ct;
    };

    void processQueue();

    CPUDevice* device;

    // Queue for executing functions asynchronously
    std::queue<Task> queue;                    // queue of tasks to execute
    bool queueShutdown = false;                // flag to signal the queue thread to shutdown
    std::thread queueThread;                   // thread that processes the queue
    std::mutex queueMutex;                     // mutex for the queue
    std::condition_variable queueCond;         // condition variable for the queue

    std::shared_ptr<tbb::task_arena> arena;    // task arena where the functions are executed
    std::shared_ptr<PinningObserver> observer; // task scheduler observer for pinning threads
    std::shared_ptr<ThreadAffinity> affinity;  // thread affinity manager for pinning threads
  };

OIDN_NAMESPACE_END
