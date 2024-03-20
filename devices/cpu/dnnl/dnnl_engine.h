// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../cpu_engine.h"
#include "mkl-dnn/include/dnnl.hpp"

OIDN_NAMESPACE_BEGIN

  class DNNLEngine final : public CPUEngine
  {
  public:
    DNNLEngine(CPUDevice* device, int numThreads);

    oidn_inline dnnl::engine& getDNNLEngine() { return dnnlEngine; }
    oidn_inline dnnl::stream& getDNNLStream() { return dnnlStream; }

    Ref<Tensor> newTensor(const TensorDesc& desc, Storage storage) override;
    Ref<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset) override;

    // Ops
    Ref<Conv> newConv(const ConvDesc& desc) override;

  private:
    dnnl::engine dnnlEngine;
    dnnl::stream dnnlStream;
  };

OIDN_NAMESPACE_END
