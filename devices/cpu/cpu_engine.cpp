// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_engine.h"
#if !defined(OIDN_DNNL) && !defined(OIDN_BNNS)
  #include "cpu_conv.h"
#endif
#include "cpu_pool.h"
#include "cpu_upsample.h"
#include "cpu_autoexposure.h"
#include "cpu_input_process.h"
#include "cpu_output_process.h"
#include "cpu_image_copy.h"

OIDN_NAMESPACE_BEGIN

  CPUEngine::CPUEngine(CPUDevice* device, int numThreads)
    : device(device)
  {
    // Get the thread affinities for one thread per core on non-hybrid CPUs with SMT
  #if !(defined(__APPLE__) && defined(OIDN_ARCH_ARM64))
    if (device->setAffinity
      #if TBB_INTERFACE_VERSION >= 12020 // oneTBB 2021.2 or later
        && tbb::info::core_types().size() <= 1 // non-hybrid cores
      #endif
       )
    {
      affinity = std::make_shared<ThreadAffinity>(1, device->verbose);
      if (affinity->getNumThreads() == 0 ||                                           // detection failed
          tbb::this_task_arena::max_concurrency() == affinity->getNumThreads() ||     // no SMT
          (tbb::this_task_arena::max_concurrency() % affinity->getNumThreads()) != 0) // hybrid SMT
        affinity.reset(); // disable affinitization
    }
  #endif

    // Create the task arena
    const int maxNumThreads = affinity ? affinity->getNumThreads() : tbb::this_task_arena::max_concurrency();
    numThreads = (numThreads > 0) ? min(numThreads, maxNumThreads) : maxNumThreads;
    arena = std::make_shared<tbb::task_arena>(numThreads);

    // Automatically set the thread affinities
    if (affinity)
      observer = std::make_shared<PinningObserver>(affinity, *arena);

    // Start the queue processing thread
    queueThread = std::thread([&]() { processQueue(); });
  }

  CPUEngine::~CPUEngine()
  {
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      queueShutdown = true;
    }
    queueCond.notify_all();
    queueThread.join();

    if (observer)
      observer.reset();
  }

#if !defined(OIDN_DNNL) && !defined(OIDN_BNNS)
  Ref<Conv> CPUEngine::newConv(const ConvDesc& desc)
  {
    return makeRef<CPUConv>(this, desc);
  }
#endif

  Ref<Pool> CPUEngine::newPool(const PoolDesc& desc)
  {
    return makeRef<CPUPool>(this, desc);
  }

  Ref<Upsample> CPUEngine::newUpsample(const UpsampleDesc& desc)
  {
    return makeRef<CPUUpsample>(this, desc);
  }

  Ref<Autoexposure> CPUEngine::newAutoexposure(const ImageDesc& srcDesc)
  {
    return makeRef<CPUAutoexposure>(this, srcDesc);
  }

  Ref<InputProcess> CPUEngine::newInputProcess(const InputProcessDesc& desc)
  {
    return makeRef<CPUInputProcess>(this, desc);
  }

  Ref<OutputProcess> CPUEngine::newOutputProcess(const OutputProcessDesc& desc)
  {
    return makeRef<CPUOutputProcess>(this, desc);
  }

  Ref<ImageCopy> CPUEngine::newImageCopy()
  {
    return makeRef<CPUImageCopy>(this);
  }

  void CPUEngine::submitFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct)
  {
    {
      std::lock_guard<std::mutex> lock(queueMutex);
      queue.push({std::move(f), ct});
    }
    queueCond.notify_all();
  }

  void CPUEngine::submitHostFunc(std::function<void()>&& f, const Ref<CancellationToken>& ct)
  {
    submitFunc(std::move(f), ct);
  }

  void CPUEngine::wait()
  {
    {
      std::unique_lock<std::mutex> lock(queueMutex);
      queueCond.wait(lock, [&] { return queue.empty(); });
    }
  }

  void* CPUEngine::usmAlloc(size_t byteSize, Storage storage)
  {
    if (storage != Storage::Host && storage != Storage::Device && storage != Storage::Managed)
      throw Exception(Error::InvalidArgument, "invalid storage mode");

    if (byteSize == 0)
      return nullptr;
    return alignedMalloc(byteSize);
  }

  void CPUEngine::usmFree(void* ptr, Storage storage)
  {
    if (ptr != nullptr)
      alignedFree(ptr);
  }

  void CPUEngine::usmCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    submitUSMCopy(dstPtr, srcPtr, byteSize);
    wait();
  }

  void CPUEngine::submitUSMCopy(void* dstPtr, const void* srcPtr, size_t byteSize)
  {
    submitFunc([=] { std::memcpy(dstPtr, srcPtr, byteSize); });
  }

  void CPUEngine::processQueue()
  {
    for (; ;)
    {
      Task task;

      // Wait until a task is available in the queue
      {
        std::unique_lock<std::mutex> lock(queueMutex);
        queueCond.wait(lock, [&] { return !queue.empty() || queueShutdown; });
        if (queue.empty() && queueShutdown)
          return;
        task = std::move(queue.front());
      }

      // Execute queued tasks in the arena until the queue gets empty
      arena->execute([&]
      {
        for (; ;)
        {
          if (task.ct && task.ct->isCancelled())
            device->setAsyncError(Error::Cancelled, "execution was cancelled");
          else
            task.func();

          task = {};

          {
            std::lock_guard<std::mutex> lock(queueMutex);
            queue.pop();
            if (queue.empty())
              break;
            task = std::move(queue.front());
          }
        }

        queueCond.notify_all();
      });
    }
  }

OIDN_NAMESPACE_END
