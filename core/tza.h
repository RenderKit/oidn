// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "tensor.h"
#include <map>

namespace oidn {

  // Parses tensors from a Tensor Archive (TZA)
  std::map<std::string, std::shared_ptr<Tensor>> parseTZA(const Ref<Device>& device, void* buffer, size_t size);

} // namespace oidn