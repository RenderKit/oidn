// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "dnnl_op.h"

namespace oidn {

  class DNNLPool : public DNNLOp, public Pool
  {
  public:
    DNNLPool(const Ref<DNNLDevice>& device, const PoolDesc& desc);
  };

} // namespace oidn
