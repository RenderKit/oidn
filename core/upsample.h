// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "op.h"

namespace oidn {

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
    virtual void setSrc(const std::shared_ptr<Tensor>& src);
    virtual void setDst(const std::shared_ptr<Tensor>& dst);
    std::shared_ptr<Tensor> getDst() const { return dst; }

  protected:
    TensorDesc dstDesc;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
  };

} // namespace oidn
