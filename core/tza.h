// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"

OIDN_NAMESPACE_BEGIN

  // Parses tensors from a Tensor Archive (TZA)
  std::shared_ptr<TensorMap> parseTZA(const void* buffer, size_t size);

OIDN_NAMESPACE_END