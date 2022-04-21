// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"

namespace oidn {

  // Abstract operation class
  class Op
  {
  public:
    virtual ~Op() = default;

    // Support must be checked before getting the scratch size or running
    virtual bool isSupported() const { return true; }

    // Scratch memory
    virtual size_t getScratchByteSize() const { return 0; }
    virtual void setScratch(const std::shared_ptr<Tensor>& scratch) {}

    // Finalization is required before running
    virtual void finalize() {}

    // Runs the operation which may be asynchronous
    virtual void run() = 0;

    // Name for debugging purposes
    std::string getName() const { return name; }
    void setName(const std::string& name) { this->name = name; }

  private:
    std::string name;
  };

} // namespace oidn
