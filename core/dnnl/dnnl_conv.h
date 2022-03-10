// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../conv.h"
#include "dnnl_op.h"

namespace oidn {

  class DNNLConv final : public DNNLOp, public Conv
  {
  public:
    DNNLConv(const Ref<DNNLDevice>& device, const ConvDesc& desc);

    size_t getScratchByteSize() const override;
    
    void setSrc(const std::shared_ptr<Tensor>& src) override;
    void setDst(const std::shared_ptr<Tensor>& dst) override;

    void finalize() override;

  private:
    dnnl::convolution_forward::primitive_desc primDesc;
  };

} // namespace oidn
