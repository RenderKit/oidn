// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "thread.h"

#define TBB_PREVIEW_LOCAL_OBSERVER 1
#include "tbb/task_scheduler_observer.h"
#include "tbb/task_arena.h"
#include "tbb/parallel_for.h"
#include "tbb/parallel_reduce.h"
#include "tbb/blocked_range.h"
#include "tbb/blocked_range2d.h"

namespace oidn {

  // ---------------------------------------------------------------------------
  // PinningObserver
  // ---------------------------------------------------------------------------

  class PinningObserver : public tbb::task_scheduler_observer
  {
  private:
    std::shared_ptr<ThreadAffinity> affinity;

  public:
    explicit PinningObserver(const std::shared_ptr<ThreadAffinity>& affinity);
    PinningObserver(const std::shared_ptr<ThreadAffinity>& affinity, tbb::task_arena& arena);
    ~PinningObserver();

    void on_scheduler_entry(bool isWorker) override;
    void on_scheduler_exit(bool isWorker) override;
  };

} // namespace oidn
