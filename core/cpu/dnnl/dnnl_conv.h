// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "dnnl_common.h"

namespace oidn {

  class DNNLConv final : public Conv
  {
  public:
    DNNLConv(const Ref<DNNLEngine>& engine, const ConvDesc& desc);

    size_t getScratchByteSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;

    void finalize() override;
    void submit() override;

  private:
    void updateSrc() override;
    void updateWeight() override;
    void updateBias() override;
    void updateDst() override;

    Ref<DNNLEngine> engine;
    dnnl::convolution_forward::primitive_desc primDesc;
    dnnl::convolution_forward prim;
    std::unordered_map<int, dnnl::memory> args;
    std::shared_ptr<Tensor> scratch;
  };

} // namespace oidn
