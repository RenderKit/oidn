// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "dnnl_common.h"

OIDN_NAMESPACE_BEGIN

  class DNNLConv final : public Conv
  {
  public:
    DNNLConv(DNNLEngine* engine, const ConvDesc& desc);

    size_t getScratchByteSize() const override;
    void setScratch(const Ref<Buffer>& scratch) override;

    void finalize() override;
    void submit() override;

  private:
    void updateSrc() override;
    void updateWeight() override;
    void updateBias() override;
    void updateDst() override;

    DNNLEngine* engine;
    dnnl::convolution_forward::primitive_desc primDesc;
    dnnl::convolution_forward prim;
    std::unordered_map<int, dnnl::memory> args;
    Ref<Buffer> scratch;
  };

OIDN_NAMESPACE_END
