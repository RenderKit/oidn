// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "progress.h"

namespace oidn {

  Progress::Progress()
    : enabled(false),
      cancelled(false),
      func(nullptr),
      userPtr(nullptr),
      total(0),
      current(0)
  {}

  void Progress::start(const Ref<Engine>& engine, ProgressMonitorFunction func, void* userPtr, double total)
  {
    cancelled = false;
    enabled = func != nullptr;
    if (!enabled)
      return;

    engine->submitHostFunc([=]()
    {
      std::lock_guard<std::mutex> lock(mutex);

      this->func = func;
      this->userPtr = userPtr;
      this->total = total;
      this->current = 0;

      call();
    });

    checkCancelled();
  }

  void Progress::update(const Ref<Engine>& engine, double done)
  {
    assert(done >= 0);

    if (!enabled)
      return;

    engine->submitHostFunc([=]()
    {
      std::lock_guard<std::mutex> lock(mutex);
      current = std::min(current + done, total);
      call();
    });

    checkCancelled();
  }

  void Progress::finish(const Ref<Engine>& engine)
  {
    if (!enabled)
      return;

    engine->submitHostFunc([=]()
    {
      std::lock_guard<std::mutex> lock(mutex);

      // Make sure total progress is reported at the end
      if (current < total)
      {
        current = total;
        call();
      }

      func = nullptr; // do not call the function anymore
    });
  }

  void Progress::call()
  {
    if (!func)
      return;
      
    if (!func(userPtr, current / total))
    {
      cancelled = true;
      func = nullptr; // do not call the function anymore, more updates could be already queued
    }
  }

  void Progress::checkCancelled()
  {
    if (cancelled)
      throw Exception(Error::Cancelled, "execution was cancelled");
  }

} // namespace oidn
