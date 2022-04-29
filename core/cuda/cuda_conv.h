// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "cuda_device.h"

namespace oidn {

  std::shared_ptr<Conv> newCUDAConv(const Ref<CUDADevice>& device, const ConvDesc& desc);

} // namespace oidn
