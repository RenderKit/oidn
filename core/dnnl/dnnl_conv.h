// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "dnnl_op.h"

namespace oidn {

  class DNNLConv : public DNNLOp, public Conv
  {
  public:
    DNNLConv(const Ref<DNNLDevice>& device, const ConvDesc& desc);
  };

} // namespace oidn
