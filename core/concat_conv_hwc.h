// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_conv.h"

OIDN_NAMESPACE_BEGIN

  // Concatenation + convolution for HWC tensors
  // The convolution is split into two smaller convolutions, one for each input tensor
  // The weights for each convolution must be set separately
  class ConcatConvHWC final : public ConcatConv
  {
  public:
    ConcatConvHWC(const Ref<Engine>& engine, const ConcatConvDesc& desc);

    bool isSupported() const override;

    size_t getScratchByteSize() const override;
    void setScratch(const Ref<Buffer>& scratch) override;

    TensorDesc getWeight1Desc() const { return weight1Desc; }
    TensorDesc getWeight2Desc() const { return weight2Desc; }
    void setWeight(const std::shared_ptr<Tensor>& weight1, const std::shared_ptr<Tensor>& weight2);

    void finalize() override;
    void submit() override;

  private:
    void updateSrc() override;
    void updateBias() override { conv1->setBias(bias); }
    void updateDst() override;

    TensorDesc weight1Desc;
    TensorDesc weight2Desc;

    std::shared_ptr<Conv> conv1;
    std::shared_ptr<Conv> conv2;

    Ref<Engine> engine;
  };

OIDN_NAMESPACE_END
