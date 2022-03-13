// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_pool.h"

namespace oidn {

  DNNLPool::DNNLPool(const Ref<DNNLDevice>& device, const PoolDesc& desc)
    : DNNLOp(device),
      Pool(desc)
  {
    const dnnl::memory::dims kernel  = {2, 2};
    const dnnl::memory::dims strides = {2, 2};
    const dnnl::memory::dims padding = {0, 0};

    auto poolDesc = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max,
      toDNNL(srcDesc),
      toDNNL(dstDesc),
      strides, kernel, padding, padding);

    dnnl::primitive_attr poolAttr;
    poolAttr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    primDesc = dnnl::pooling_forward::primitive_desc(poolDesc, poolAttr, device->getDNNLEngine());

    prim = dnnl::pooling_forward(primDesc);
  }

  size_t DNNLPool::getScratchByteSize() const
  {
    return primDesc.scratchpad_desc().get_size();
  }

  void DNNLPool::setSrc(const std::shared_ptr<Tensor>& src)
  {
    Pool::setSrc(src);
    args[DNNL_ARG_SRC] = getDNNL(src);
  }

  void DNNLPool::setDst(const std::shared_ptr<Tensor>& dst)
  {
    Pool::setDst(dst);
    args[DNNL_ARG_DST] = getDNNL(dst);
  }

  void DNNLPool::finalize()
  {
    prim = dnnl::pooling_forward(primDesc);
  }

} // namespace oidn