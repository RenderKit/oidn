// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.h"

OIDN_NAMESPACE_BEGIN

  // Abstract operation class
  class Op : public RefCount
  {
  public:
    virtual ~Op() = default;

    // Support must be checked before getting the scratch size or running
    virtual bool isSupported() const { return true; }

    // Scratch memory
    virtual size_t getScratchByteSize() const { return 0; }
    virtual void setScratch(const Ref<Buffer>& scratch) {}

    // Finalization is required before running
    virtual void finalize() {}

    // Runs the operation which may be asynchronous
    virtual void submit()
    {
      throw std::logic_error("operation is not implemented");
    }

    // Name for debugging purposes
    std::string getName() const { return name; }
    void setName(const std::string& name) { this->name = name; }

  private:
    std::string name;
  };

OIDN_NAMESPACE_END
