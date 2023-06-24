// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/op.h"
#include <vector>

OIDN_NAMESPACE_BEGIN

  enum class MetalOpType
  {
    Input,
    Conv,
    Add,
    Relu,
    Pool,
    Concat,
    Upsample,
    Output
  };

  class MetalOp : public Op
  {
  public:
    MetalOp(MetalOpType opType, std::vector<std::shared_ptr<Op>> srcs);
    
    MetalOpType getOpType() {return opType;}
    const std::vector<std::shared_ptr<Op>>& getSrc() {return srcs;}
    
    void submit() override {throw std::logic_error("MetalOp can not be submitted");}
    
  protected:
    MetalOpType opType;
    std::vector<std::shared_ptr<Op>> srcs;
  };

OIDN_NAMESPACE_END
