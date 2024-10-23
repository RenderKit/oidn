// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "engine.h"
#include "buffer.h"

OIDN_NAMESPACE_BEGIN

  // Abstract operation class
  class Op : public RefCount
  {
  public:
    virtual ~Op() = default;

    virtual Engine* getEngine() const = 0;

    // Support must be checked before getting the scratch size or submission
    virtual bool isSupported() const { return true; }

    // Scratch memory
    virtual size_t getScratchByteSize() { return 0; }
    virtual void setScratch(const Ref<Buffer>& scratch) {}

    // Finalization is required before submission
    virtual void finalize() {}

    // Enqueues the operation to the engine, optionally updating the progress as well
    virtual void submit(const Ref<Progress>& progress = nullptr) = 0;

    // Returns the estimated amount of work for progress monitoring
    virtual size_t getWorkAmount() const { return 1; }

    // Name for debugging purposes
    std::string getName() const { return name; }
    void setName(const std::string& name) { this->name = name; }

  private:
    std::string name;
  };

  // Base class for most operations (except compound operations, e.g. Graph)
  class BaseOp : public Op
  {
  public:
    void submit(const Ref<Progress>& progress) final;

    // Enqueues the kernel(s) of the operation to the engine, which may be cancelled if supported
    virtual void submitKernels(const Ref<CancellationToken>& ct = nullptr) = 0;
  };

OIDN_NAMESPACE_END
