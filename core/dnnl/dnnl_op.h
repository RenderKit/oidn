// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../op.h"
#include "dnnl_device.h"
#include "dnnl_tensor.h"

namespace oidn {

  // DNNL operation base class
  class DNNLOp : public BaseOp<DNNLDevice>
  {
  public:
    DNNLOp(const Ref<DNNLDevice>& device);
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;
    void run() override;

  protected:
    dnnl::primitive prim;
    std::unordered_map<int, dnnl::memory> args;
    std::shared_ptr<Tensor> scratch;
  };

} // namespace oidn
