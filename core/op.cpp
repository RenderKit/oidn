// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "op.h"

OIDN_NAMESPACE_BEGIN

  void BaseOp::submit(const Ref<Progress>& progress)
  {
    Engine* engine = nullptr;

    if (progress)
    {
      engine = getEngine();
      Progress::submitUpdate(engine, progress);
    }

    submitKernels(progress);

    if (progress)
      Progress::submitUpdate(engine, progress, getWorkAmount());
  }

OIDN_NAMESPACE_END