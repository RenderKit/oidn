// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#include <chrono>

OIDN_NAMESPACE_BEGIN

  class Timer
  {
  public:
    Timer()
    {
      reset();
    }

    void reset()
    {
      start = clock::now();
    }

    double query() const
    {
      auto end = clock::now();
      return std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    }

  private:
    using clock = std::chrono::steady_clock;

    std::chrono::time_point<clock> start;
  };

OIDN_NAMESPACE_END
