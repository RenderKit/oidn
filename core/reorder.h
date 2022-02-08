// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

namespace oidn {

  struct ReorderDesc
  {
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

} // namespace oidn
