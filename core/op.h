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

    virtual Device* getDevice() const = 0;

    // Support must be checked before getting the scratch size or running
    virtual bool isSupported() const { return true; }

    // Scratch memory
    virtual size_t getScratchByteSize() const { return 0; }
    virtual void setScratch(const std::shared_ptr<Tensor>& scratch) {}

    // Name for debugging purposes
    virtual std::string getName() const = 0;
    virtual void setName(const std::string& name) = 0;

    // Finalization is required before running
    virtual void finalize() {}

    virtual void run() = 0;
  };

  // Operation base class
  template<typename DeviceT = Device>
  class BaseOp : public virtual Op
  {
  public:
    using DeviceType = DeviceT;

    BaseOp(const Ref<DeviceT>& device)
      : device(device) {}

    Device* getDevice() const override { return device.get(); }

    std::string getName() const override { return name; }
    void setName(const std::string& name) override { this->name = name; }

  protected:
    Ref<DeviceT> device;
    std::string name;
  };

} // namespace oidn
