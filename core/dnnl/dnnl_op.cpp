// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_op.h"

namespace oidn {

  DNNLOp::DNNLOp(const Ref<DNNLDevice>& device)
    : BaseOp(device) {}

  void DNNLOp::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
    args[DNNL_ARG_SCRATCHPAD] = getDNNL(scratch);
  }

  void DNNLOp::run()
  {
    prim.execute(device->getDNNLStream(), args);
  }

} // namespace oidn
