// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"

namespace oidn {

  // Abstract node class
  class Node
  {
  public:
    virtual ~Node() = default;

    virtual void execute() = 0;

    virtual size_t getScratchSize() const { return 0; }
    virtual void setScratch(const std::shared_ptr<Tensor>& scratch) {}

    virtual Device* getDevice() = 0;
    virtual std::string getName() const = 0;
  };

  // Node base class
  template<typename DeviceT>
  class BaseNode : public virtual Node
  {
  protected:
    Ref<DeviceT> device;
    std::string name;

  public:
    using DeviceType = DeviceT;

    BaseNode(const Ref<DeviceT>& device, const std::string& name)
      : device(device),
        name(name) {}

    Device* getDevice() override { return device.get(); }
    std::string getName() const override { return name; }
  };

} // namespace oidn
