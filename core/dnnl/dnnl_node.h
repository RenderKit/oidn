// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../node.h"
#include "dnnl_device.h"
#include "dnnl_tensor.h"

namespace oidn {

  // DNNL node base class
  class DNNLNode : public BaseNode<DNNLDevice>
  {
  protected:
    dnnl::primitive prim;
    std::unordered_map<int, dnnl::memory> args;
    std::shared_ptr<Tensor> scratch;

  public:
    DNNLNode(const Ref<DNNLDevice>& device, const std::string& name);

    size_t getScratchSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;
    void execute() override;
  };

} // namespace oidn
