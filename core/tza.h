// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"
#include <map>

namespace oidn {

  // Parses tensors from a Tensor Archive (TZA)
  std::map<std::string, Tensor> parseTZA(void* buffer, size_t size);

} // namespace oidn