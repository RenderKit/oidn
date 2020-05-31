// Copyright 2009-2020 Intel Corporation
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

  public:
    Progress(ProgressMonitorFunction func, void* userPtr, double total = 1)
     : func(func),
       userPtr(userPtr),
       total(total),
       current(0)
    {
      update();
    }

    ~Progress()
    {
      // Make sure total progress is reported at the end
      if (current < total)
      {
        current = total;
        update();
      }
    }

    // Calls the progress monitor function
    void update()
    {
      if (func)
      {
        if (!func(userPtr, current / total))
          throw Exception(Error::Cancelled, "execution was cancelled");
      }
    }

    // Advances the progress with the specified amount and calls the progress monitor function
    void update(double done)
    {
      current = std::min(current + done, total);
      update();
    }
  };

} // namespace oidn
