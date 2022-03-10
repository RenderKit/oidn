// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../device.h"
#include "dnnl_common.h"

namespace oidn {

  class DNNLDevice : public Device
  {
  public:
    OIDN_INLINE dnnl::engine& getDNNLEngine() { return dnnlEngine; }
    OIDN_INLINE dnnl::stream& getDNNLStream() { return dnnlStream; }

    void wait() override;

    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc) override;
    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, void* data) override;
    std::shared_ptr<Tensor> newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset) override;

    // Ops
    std::shared_ptr<Conv> newConv(const ConvDesc& desc) override;
    std::shared_ptr<Pool> newPool(const PoolDesc& desc) override;

  protected:
    dnnl::engine dnnlEngine;
    dnnl::stream dnnlStream;
  };

} // namespace oidn
