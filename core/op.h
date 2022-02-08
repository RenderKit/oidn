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

    virtual void run() = 0;
    virtual std::shared_ptr<Tensor> getDst() const { return nullptr; }

    virtual size_t getScratchSize() const { return 0; }
    virtual void setScratch(const std::shared_ptr<Tensor>& scratch) {}

    // Name for debugging purposes
    virtual std::string getName() const = 0;
    virtual void setName(const std::string& name) = 0;

    virtual Device* getDevice() = 0;
  };

  // Operation base class
  template<typename DeviceT>
  class BaseOp : public virtual Op
  {
  protected:
    Ref<DeviceT> device;
    std::string name;

  public:
    using DeviceType = DeviceT;

    BaseOp(const Ref<DeviceT>& device)
      : device(device) {}

    std::string getName() const override { return name; }
    void setName(const std::string& name) override { this->name = name; }

    Device* getDevice() override { return device.get(); }
  };

} // namespace oidn
