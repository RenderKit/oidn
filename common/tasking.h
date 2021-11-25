// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "thread.h"

#if defined(__clang__) && !defined(_LIBCPP_VERSION) && !defined(TBB_USE_GLIBCXX_VERSION)
  // TBB does not always detect the version of libstdc++ correctly when using
  // Clang, so we have to set it manually. This is required for some TBB
  // features (e.g. TBB_USE_CAPTURED_EXCEPTION=0).
  #if defined(_GLIBCXX_RELEASE)
    // This macro is available only in GCC 7.1 and later
    #define TBB_USE_GLIBCXX_VERSION (_GLIBCXX_RELEASE * 10000)
  #else
    // Try to detect some older GCC versions
    #if __has_include(<cuchar>)
      #define TBB_USE_GLIBCXX_VERSION 60000
    #elif __has_include(<codecvt>)
      #define TBB_USE_GLIBCXX_VERSION 50000
    #elif __has_include(<ext/cmath>)
      #define TBB_USE_GLIBCXX_VERSION 40800
    #endif
  #endif
#endif

#define TBB_USE_CAPTURED_EXCEPTION 0
#define TBB_PREVIEW_LOCAL_OBSERVER 1
#define TBB_PREVIEW_TASK_ARENA_CONSTRAINTS_EXTENSION 1

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
