// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "tensor.h"

OIDN_NAMESPACE_BEGIN

  struct UpsampleDesc
  {
    TensorDesc srcDesc;
  };

  // 2x2 nearest-neighbor upsampling
  class Upsample : public Op, protected UpsampleDesc
  {
  public:
    Upsample(const UpsampleDesc& desc);
    
    TensorDesc getDstDesc() const { return dstDesc; }
    std::shared_ptr<Tensor> getDst() const { return dst; }

    void setSrc(const std::shared_ptr<Tensor>& src);
    void setDst(const std::shared_ptr<Tensor>& dst);

  protected:
    virtual void updateSrc() {}
    virtual void updateDst() {}

    TensorDesc dstDesc;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

OIDN_NAMESPACE_END
