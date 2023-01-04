// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../cpu/cpu_engine.h"

namespace oidn {

  class DNNLEngine final : public CPUEngine
  {
  public:
    explicit DNNLEngine(const Ref<CPUDevice>& device);

    OIDN_INLINE dnnl::engine& getDNNLEngine() { return dnnlEngine; }
    OIDN_INLINE dnnl::stream& getDNNLStream() { return dnnlStream; }

    void wait() override;

    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, Storage storage) override;
    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, void* data) override;
    std::shared_ptr<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset) override;

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;

  private:
    dnnl::engine dnnlEngine;
    dnnl::stream dnnlStream;
  };

} // namespace oidn
