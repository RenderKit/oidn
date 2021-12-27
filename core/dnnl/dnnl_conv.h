// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "dnnl_node.h"

namespace oidn {

  class DNNLConvNode : public DNNLNode, public ConvNode
  {
  public:
    DNNLConvNode(const Ref<DNNLDevice>& device, const ConvDesc& desc);
  };

} // namespace oidn
