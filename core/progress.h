// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine.h"

OIDN_NAMESPACE_BEGIN

  // Asynchronous progress monitoring
  class Progress
  {
  public:
    Progress();

    // Starts progress monitoring using the specified callback function
    void start(Engine* engine, ProgressMonitorFunction func, void* userPtr, double total = 1);

    // Advances the progress with the specified amount and calls the progress monitor function
    void update(Engine* engine, double done);

    // Finishes monitoring, setting the progress to the total value
    void finish(Engine* engine);

  private:
    // Calls the progress monitor function
    void call();

    // Checks whether cancellation has been requested
    void checkCancelled();

    bool enabled; // is progress monitoring currently enabled?
    std::atomic<bool> cancelled; // has cancellation been requested by the callback?

    // Asynchronous progress state
    ProgressMonitorFunction func;
    void* userPtr;
    double total;   // maximum progress value
    double current; // current progress value

    std::mutex mutex; // for thread safety
  };

OIDN_NAMESPACE_END
