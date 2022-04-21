// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../pool.h"
#include "dnnl_common.h"

namespace oidn {

  class DNNLPool final : public Pool
  {
  public:
    DNNLPool(const Ref<DNNLDevice>& device, const PoolDesc& desc);

    size_t getScratchByteSize() const override;
    void setScratch(const std::shared_ptr<Tensor>& scratch) override;
    
    void setSrc(const std::shared_ptr<Tensor>& src) override;
    void setDst(const std::shared_ptr<Tensor>& dst) override;

    void finalize() override;
    void run() override;

  private:
    Ref<DNNLDevice> device;
    dnnl::pooling_forward::primitive_desc primDesc;
    dnnl::pooling_forward prim;
    std::unordered_map<int, dnnl::memory> args;
    std::shared_ptr<Tensor> scratch;
  };

} // namespace oidn
