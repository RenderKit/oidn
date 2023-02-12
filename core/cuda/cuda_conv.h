// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "cuda_engine.h"

OIDN_NAMESPACE_BEGIN

  std::shared_ptr<Conv> newCUDAConv(const Ref<CUDAEngine>& engine, const ConvDesc& desc);

OIDN_NAMESPACE_END
