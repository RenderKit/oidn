// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../concat_conv.h"
#include "cuda_conv.h"

namespace oidn {

  class CUDAConcatConv final : public ConcatConv
  {
  public:
    CUDAConcatConv(const Ref<CUDAEngine>& engine, const ConcatConvDesc& desc);

    bool isSupported() const override;

    size_t getScratchByteSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;

    void finalize() override;
    void submit() override;

  private:
    void updateSrc() override;
    void updateWeight() override;
    void updateBias() override;
    void updateDst() override;

    Ref<CUDAEngine> engine;
    bool finalized = false;

    TensorDesc weight1Desc;
    TensorDesc weight2Desc;

    std::shared_ptr<Conv> conv1;
    std::shared_ptr<Conv> conv2;
  };

} // namespace oidn
