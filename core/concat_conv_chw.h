// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "concat_conv.h"

OIDN_NAMESPACE_BEGIN

  // Concatenation + convolution for CHW tensors (including blocked) stored consecutively in memory
  // Since the tensors are pre-concatenated in memory, only the convolution needs to be executed
  class ConcatConvCHW final : public ConcatConv
  {
  public:
    ConcatConvCHW(const Ref<Engine>& engine, const ConcatConvDesc& desc);

    size_t getScratchByteSize() const override { return conv->getScratchByteSize(); }
    void setScratch(const Ref<Buffer>& scratch) override { conv->setScratch(scratch); }

    void setWeight(const Ref<Tensor>& weight) { conv->setWeight(weight); }

    void finalize() override { conv->finalize(); }
    void submit() override { conv->submit(); }

  private:
    void updateSrc() override;
    void updateBias() override { conv->setBias(bias); }
    void updateDst() override { conv->setDst(dst); }

    TensorDesc srcDesc;         // pre-concatenated source
    std::shared_ptr<Conv> conv;
    Ref<Engine> engine;
  };

OIDN_NAMESPACE_END