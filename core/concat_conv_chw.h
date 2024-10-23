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
    ConcatConvCHW(Engine* engine, const ConcatConvDesc& desc);

    Engine* getEngine() const override { return conv->getEngine(); }

    size_t getScratchByteSize() override { return conv->getScratchByteSize(); }
    void setScratch(const Ref<Buffer>& scratch) override { conv->setScratch(scratch); }

    void setWeight(const Ref<Tensor>& weight) { conv->setWeight(weight); }

    void finalize() override { conv->finalize(); }
    void submitKernels(const Ref<CancellationToken>& ct) override { conv->submitKernels(ct); }

  private:
    void updateSrc() override;
    void updateBias() override { conv->setBias(bias); }
    void updateDst() override { conv->setDst(dst); }

    TensorDesc srcDesc;         // pre-concatenated source
    Ref<Conv> conv;
  };

OIDN_NAMESPACE_END