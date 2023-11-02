// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tasking.h"

OIDN_NAMESPACE_BEGIN

  // -----------------------------------------------------------------------------------------------
  // PinningObserver
  // -----------------------------------------------------------------------------------------------

  PinningObserver::PinningObserver(const std::shared_ptr<ThreadAffinity>& affinity)
    : affinity(affinity)
  {
    observe(true);
  }

  PinningObserver::PinningObserver(const std::shared_ptr<ThreadAffinity>& affinity, tbb::task_arena& arena)
    : tbb::task_scheduler_observer(arena),
      affinity(affinity)
  {
    observe(true);
  }

  PinningObserver::~PinningObserver()
  {
    observe(false);
  }

  void PinningObserver::on_scheduler_entry(bool isWorker)
  {
    const int threadIndex = tbb::this_task_arena::current_thread_index();
    if (threadIndex >= 0)
      affinity->set(threadIndex);
  }

  void PinningObserver::on_scheduler_exit(bool isWorker)
  {
    const int threadIndex = tbb::this_task_arena::current_thread_index();
    if (threadIndex >= 0)
      affinity->restore(threadIndex);
  }

OIDN_NAMESPACE_END
