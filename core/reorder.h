// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

OIDN_NAMESPACE_BEGIN

  struct ReorderDesc
  {
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

  // Reorder with zero padding
  void reorder(const Tensor& src, Tensor& dst);

OIDN_NAMESPACE_END
