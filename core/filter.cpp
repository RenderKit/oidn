// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "filter.h"

namespace oidn {

  void Filter::setProgressMonitorFunction(ProgressMonitorFunction func, void* userPtr)
  {
    progressFunc = func;
    progressUserPtr = userPtr;
  }

} // namespace oidn
