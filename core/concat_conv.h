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
    std::shared_ptr<Tensor> weight;
    std::shared_ptr<Tensor> bias;
    bool relu;
  };

  class ConcatConv : public Op, protected ConcatConvDesc
  {
  public:    
    ConcatConv(const ConcatConvDesc& desc);

    TensorDesc getDstDesc() const { return dstDesc; }
    virtual void setSrc(const std::shared_ptr<Tensor>& src1, const std::shared_ptr<Tensor>& src2);
    virtual void setDst(const std::shared_ptr<Tensor>& dst);
    std::shared_ptr<Tensor> getDst() const { return dst; }

  protected:
    TensorDesc dstDesc;
    std::shared_ptr<Tensor> src1;
    std::shared_ptr<Tensor> src2;
    std::shared_ptr<Tensor> dst;
  };

  // Concatenation + 3x3 convolution for CHW tensors (including blocked) stored consecutively in memory
  // Since the tensors are pre-concatenated in memory, only the convolution needs to be executed
  class CHWConcatConv final : public ConcatConv
  {
  public:
    CHWConcatConv(const Ref<Device>& device, const ConcatConvDesc& desc);

    size_t getScratchByteSize() const override { return conv->getScratchByteSize(); }
    void setScratch(const std::shared_ptr<Tensor>& scratch) override { conv->setScratch(scratch); }

    void setSrc(const std::shared_ptr<Tensor>& src1, const std::shared_ptr<Tensor>& src2) override;
    void setDst(const std::shared_ptr<Tensor>& dst) override;

    void finalize() override { conv->finalize(); }
    void run() override { conv->run(); }

  private:
    Ref<Device> device;
    TensorDesc srcDesc; // concatenated source
    std::shared_ptr<Conv> conv;
  };

} // namespace oidn
