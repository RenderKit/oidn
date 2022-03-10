// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "dnnl_op.h"

namespace oidn {

  class DNNLPool final : public DNNLOp, public Pool
  {
  public:
    DNNLPool(const Ref<DNNLDevice>& device, const PoolDesc& desc);

    size_t getScratchByteSize() const override;
    
    void setSrc(const std::shared_ptr<Tensor>& src) override;
    void setDst(const std::shared_ptr<Tensor>& dst) override;

    void finalize() override;

  private:
    dnnl::pooling_forward::primitive_desc primDesc;
  };

} // namespace oidn
