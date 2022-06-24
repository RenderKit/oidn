// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"

namespace oidn {

  // MPS node base class
  class MPSNode : public Node
  {
  private:
    std::shared_ptr<Tensor> dst;
    
  public:
    MPSNode(const Ref<Device>& device, const std::string& name, const std::shared_ptr<Tensor>& dst)
      : Node(device, name), dst(dst) {}

    size_t getScratchSize() const override
    {
      return 0;
    }

    void setScratch(const std::shared_ptr<Tensor>& scratch) override
    {
    }

    void execute() override
    {
    }
    
    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
