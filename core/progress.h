// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common.h"

namespace oidn {

  // Progress state
  class Progress
  {
  private:
    ProgressMonitorFunction func;
    void* userPtr;
    double total;   // maximum progress value
    double current; // current progress value

    // Calls the progress monitor function
    void update()
    {
      if (func)
      {
        if (!func(userPtr, current / total))
          throw Exception(Error::Cancelled, "execution was cancelled");
      }
    }

  public:
    Progress(ProgressMonitorFunction func, void* userPtr, double total = 1)
     : func(func),
       userPtr(userPtr),
       total(total),
       current(0)
    {
      update();
    }

    // Advances the progress with the specified amount and calls the progress monitor function
    void update(double done)
    {
      assert(done >= 0);
      current = std::min(current + done, total);
      update();
    }

    void finish()
    {
      // Make sure total progress is reported at the end
      if (current < total)
      {
        current = total;
        update();
      }
    }
  };

} // namespace oidn
