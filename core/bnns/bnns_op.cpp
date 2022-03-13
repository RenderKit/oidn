// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_op.h"

namespace oidn {

  BNNSOp::BNNSOp(const Ref<BNNSDevice>& device)
    : BaseOp(device) {}

  BNNSOp::~BNNSOp()
  {
    if (filter)
      BNNSFilterDestroy(filter);
  }

} // namespace oidn