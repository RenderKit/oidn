// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/tensor.h"
#include "dnnl_common.h"

namespace oidn {
  
  // Native DNNL tensor
  class DNNLTensor final : public Tensor
  {
  public:
    DNNLTensor(const Ref<DNNLEngine>& engine, const TensorDesc& desc, Storage storage);
    DNNLTensor(const Ref<DNNLEngine>& engine, const TensorDesc& desc, void* data);
    DNNLTensor(const Ref<DNNLEngine>& engine, const dnnl::memory::desc& desc);
    DNNLTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset);

    void* getData() override { return mem.get_data_handle(); }
    const void* getData() const override { return mem.get_data_handle(); }

    const dnnl::memory& getDNNLMemory() const { return mem; }

  private:
    void init(const Ref<DNNLEngine>& engine, void* data);
    void updatePtr() override;

    dnnl::memory mem;
  };

} // namespace oidn
