// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"
#include <mutex>
#include <condition_variable>

namespace oidn {

  class Barrier
  {
  public:
    Barrier(int count) : count(count) {}

    void wait()
    {
      std::unique_lock<std::mutex> lk(m);
      count--;

      if (count == 0)
      {
        lk.unlock();
        cv.notify_all();
      }
      else
      {
        cv.wait(lk, [&]{ return count == 0; });
      }
    }

  private:
    std::mutex m;
    std::condition_variable cv;
    volatile int count;
  };

} // namespace oidn
