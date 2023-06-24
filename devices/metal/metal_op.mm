// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include "metal_op.h"

OIDN_NAMESPACE_BEGIN

  MetalOp::MetalOp(MetalOpType opType, std::vector<std::shared_ptr<Op>> srcs)
    : opType(opType), srcs(srcs) {}

OIDN_NAMESPACE_END
