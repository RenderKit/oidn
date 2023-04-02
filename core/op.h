// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.h"

OIDN_NAMESPACE_BEGIN

  // Abstract operation class
  class Op
  {
  public:
    virtual ~Op() = default;

    // Support must be checked before getting the scratch size or running
    virtual bool isSupported() const { return true; }

    // Scratch memory
    virtual size_t getScratchByteSize() const { return 0; }
    size_t getScratchAlignedSize() const { return round_up(getScratchByteSize(), memoryAlignment); }
    virtual void setScratch(const Ref<Buffer>& scratch) {}

    // Finalization is required before running
    virtual void finalize() {}

    // Runs the operation whicch may be asynchronous
    virtual void submit() = 0;

    // Name for debugging purposes
    std::string getName() const { return name; }
    void setName(const std::string& name) { this->name = name; }

  private:
    std::string name;
  };

OIDN_NAMESPACE_END
