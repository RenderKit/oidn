// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"
#include "tensor.h"

OIDN_NAMESPACE_BEGIN

  // 2x2 max pooling descriptor
  struct PoolDesc
  {
    TensorDesc srcDesc;
  };

  // 2x2 max pooling
  class Pool : public BaseOp, protected PoolDesc
  {
  public:
    Pool(const PoolDesc& desc);

    TensorDesc getDstDesc() const { return dstDesc; }
    Ref<Tensor> getDst() const { return dst; }

    void setSrc(const Ref<Tensor>& src);
    void setDst(const Ref<Tensor>& dst);

  protected:
    virtual void updateSrc() {}
    virtual void updateDst() {}

    TensorDesc dstDesc;
    Ref<Tensor> src;
    Ref<Tensor> dst;
  };

OIDN_NAMESPACE_END
