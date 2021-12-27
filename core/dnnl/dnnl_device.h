// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"
#include "mkl-dnn/include/dnnl.hpp"

namespace oidn {

  class DNNLDevice : public Device
  {
  protected:
    dnnl::engine dnnlEngine;
    dnnl::stream dnnlStream;

  public:
    OIDN_INLINE dnnl::engine& getDNNLEngine() { return dnnlEngine; }
    OIDN_INLINE dnnl::stream& getDNNLStream() { return dnnlStream; }

    void wait() override;

    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc) override;
    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, void* data) override;
    std::shared_ptr<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset) override;

    // Nodes
    std::shared_ptr<ConvNode> newConvNode(const ConvDesc& desc) override;
    std::shared_ptr<PoolNode> newPoolNode(const PoolDesc& desc) override;
  };

} // namespace oidn
