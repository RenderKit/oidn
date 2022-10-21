// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "conv.h"

namespace oidn {

  // Concatenation + 3x3 convolution descriptor
  struct ConcatConvDesc
  {
    TensorDesc src1Desc;
    TensorDesc src2Desc;
    TensorDesc weightDesc;
    TensorDesc biasDesc;
    Activation activation;
  };

  class ConcatConv : public Op, protected ConcatConvDesc
  {
  public:    
    ConcatConv(const ConcatConvDesc& desc);

    TensorDesc getDstDesc() const { return dstDesc; }
    std::shared_ptr<Tensor> getDst() const { return dst; }

    void setSrc(const std::shared_ptr<Tensor>& src1, const std::shared_ptr<Tensor>& src2);
    void setWeight(const std::shared_ptr<Tensor>& weight);
    void setBias(const std::shared_ptr<Tensor>& bias);
    void setDst(const std::shared_ptr<Tensor>& dst);

  protected:
    virtual void updateSrc() {}
    virtual void updateWeight() {}
    virtual void updateBias() {}
    virtual void updateDst() {}

    TensorDesc dstDesc;
    std::shared_ptr<Tensor> src1;
    std::shared_ptr<Tensor> src2;
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    std::shared_ptr<Tensor> dst;
  };

  // Concatenation + 3x3 convolution for CHW tensors (including blocked) stored consecutively in memory
  // Since the tensors are pre-concatenated in memory, only the convolution needs to be executed
  class CHWConcatConv final : public ConcatConv
  {
  public:
    CHWConcatConv(const Ref<Engine>& engine, const ConcatConvDesc& desc);

    size_t getScratchByteSize() const override { return conv->getScratchByteSize(); }
    void setScratch(const std::shared_ptr<Tensor>& scratch) override { conv->setScratch(scratch); }

    void finalize() override { conv->finalize(); }
    void submit() override { conv->submit(); }

  private:
    void updateSrc() override;
    void updateWeight() override { conv->setWeight(weight); }
    void updateBias() override { conv->setBias(bias); }
    void updateDst() override { conv->setDst(dst); }

    Ref<Engine> engine;
    TensorDesc srcDesc; // concatenated source
    std::shared_ptr<Conv> conv;
  };

} // namespace oidn
