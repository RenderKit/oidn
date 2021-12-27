// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_pool.h"

namespace oidn {

  DNNLPoolNode::DNNLPoolNode(const Ref<DNNLDevice>& device, const PoolDesc& desc)
    : DNNLNode(device, desc.name),
      PoolNode(desc)
  {
    const dnnl::memory::dims kernel  = {2, 2};
    const dnnl::memory::dims strides = {2, 2};
    const dnnl::memory::dims padding = {0, 0};

    const dnnl::memory& srcMem = DNNLTensor::getMemory(*src);
    const dnnl::memory& dstMem = DNNLTensor::getMemory(*dst);

    auto poolDesc = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max,
      srcMem.get_desc(),
      dstMem.get_desc(),
      strides, kernel, padding, padding);

    dnnl::primitive_attr poolAttr;
    poolAttr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto poolPrimDesc = dnnl::pooling_forward::primitive_desc(poolDesc, poolAttr, device->getDNNLEngine());

    prim = dnnl::pooling_forward(poolPrimDesc);
    args = {{DNNL_ARG_SRC, srcMem},
            {DNNL_ARG_DST, dstMem}};
  }

} // namespace oidn