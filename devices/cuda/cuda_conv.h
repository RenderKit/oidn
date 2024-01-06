// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "cuda_engine.h"

OIDN_NAMESPACE_BEGIN

  Ref<Conv> newCUDAConv(CUDAEngine* engine, const ConvDesc& desc);

OIDN_NAMESPACE_END
