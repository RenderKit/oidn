// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"
#include "ref.h"
#include <mutex>

OIDN_NAMESPACE_BEGIN

  class Engine;

  // Cancellation request state for asynchronous operations
  class CancellationToken : public RefCount
  {
  public:
    CancellationToken() : cancelled(false) {}

    bool isCancelled() const { return cancelled; }
    void cancel() { cancelled = true; }

  protected:
    std::atomic<bool> cancelled;
  };

  // Progress monitoring for asynchronous operations
  class Progress : public CancellationToken
  {
  public:
    Progress(ProgressMonitorFunction func, void* userPtr, size_t total);

    // Enqueues a progress update, advancing the progress with the specified amount, and calling
    // the progress monitor function
    static void submitUpdate(Engine* engine, const Ref<Progress>& progress, size_t delta = 0);

  private:
    ProgressMonitorFunction func;
    void* userPtr;
    size_t total;     // maximum progress value
    size_t current;   // current progress value
    bool started;     // whether any progress updates have been submitted yet
    std::mutex mutex;

    void update(size_t delta);
  };

OIDN_NAMESPACE_END
