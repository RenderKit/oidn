// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

OIDN_NAMESPACE_BEGIN

  // 2x2 max pooling descriptor
  struct PoolDesc
  {
    TensorDesc srcDesc;
  };

  // 2x2 max pooling
  class Pool : public Op, protected PoolDesc
  {
  public:
    Pool(const PoolDesc& desc);
    
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
