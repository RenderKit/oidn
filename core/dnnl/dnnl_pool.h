// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "dnnl_node.h"

namespace oidn {

  class DNNLPoolNode : public DNNLNode, public PoolNode
  {
  public:
    DNNLPoolNode(const Ref<DNNLDevice>& device, const PoolDesc& desc);
  };

} // namespace oidn
