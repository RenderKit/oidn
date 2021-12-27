// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../tensor.h"
#include "dnnl_device.h"

namespace oidn {
  
  // Native DNNL tensor
  class DNNLTensor : public Tensor
  {
  private:
    dnnl::memory mem;

  public:
    DNNLTensor(const Ref<DNNLDevice>& device, const TensorDesc& desc);
    DNNLTensor(const Ref<DNNLDevice>& device, const TensorDesc& desc, void* data);
    DNNLTensor(const Ref<DNNLDevice>& device, const dnnl::memory::desc& desc);
    DNNLTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset);

    void* data() override { return mem.get_data_handle(); }
    const void* data() const override { return mem.get_data_handle(); }

    // Returns the internal DNNL memory structure of a tensor
    static const dnnl::memory& getMemory(const Tensor& tz);

  private:
    void init(const Ref<DNNLDevice>& device);
    void init(const Ref<DNNLDevice>& device, void* data);
    void updatePtr() override;

    static dnnl::memory::desc toMemoryDesc(const TensorDesc& tz);
  };

} // namespace oidn
