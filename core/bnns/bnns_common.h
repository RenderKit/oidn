// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <Accelerate/Accelerate.h>
#include "../tensor.h"
#include "bnns_device.h"

namespace oidn {

  BNNSNDArrayDescriptor toBNNS(const TensorDesc& td);
  BNNSNDArrayDescriptor toBNNS(const std::shared_ptr<Tensor>& t);

} // namespace oidn