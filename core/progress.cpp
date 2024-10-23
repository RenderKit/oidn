// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "progress.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  Progress::Progress(ProgressMonitorFunction func, void* userPtr, size_t total)
    : func(func),
      userPtr(userPtr),
      total(total),
      current(0),
      started(false)
  {
    if (!func)
      throw std::invalid_argument("progress monitor function is null");
  }

  void Progress::update(size_t delta)
  {
    std::lock_guard<std::mutex> lock(mutex);
    current = std::min(current + delta, total);
    if (!func(userPtr, double(current) / double(total)))
      cancel();
  }

  void Progress::submitUpdate(Engine* engine, const Ref<Progress>& progress, size_t delta)
  {
    if (progress->isCancelled())
      throw Exception(Error::Cancelled, "execution was cancelled");

    if (!progress->started || delta != 0) // always submit the first update
    {
      engine->submitHostFunc([progress, delta]() { progress->update(delta); }, progress);
      progress->started = true;
    }
  }

OIDN_NAMESPACE_END
