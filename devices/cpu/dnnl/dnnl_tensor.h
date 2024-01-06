// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/tensor.h"
#include "dnnl_common.h"

OIDN_NAMESPACE_BEGIN

  // Native DNNL tensor
  class DNNLTensor final : public Tensor
  {
  public:
    DNNLTensor(DNNLEngine* engine, const TensorDesc& desc, Storage storage);
    DNNLTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset);

    void* getPtr() const override { return mem.get_data_handle(); }
    const dnnl::memory& getDNNLMemory() const { return mem; }

  private:
    void postRealloc() override;

    dnnl::memory mem;
  };

OIDN_NAMESPACE_END
