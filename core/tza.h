// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"
#include <unordered_map>

OIDN_NAMESPACE_BEGIN

  // Parses tensors from a Tensor Archive (TZA)
  std::unordered_map<std::string, std::shared_ptr<Tensor>> parseTZA(const Ref<Engine>& engine, void* buffer, size_t size);

OIDN_NAMESPACE_END