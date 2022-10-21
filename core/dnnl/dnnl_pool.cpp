// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_pool.h"

namespace oidn {

  DNNLPool::DNNLPool(const Ref<DNNLEngine>& engine, const PoolDesc& desc)
    : Pool(desc),
      engine(engine)
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

    primDesc = dnnl::pooling_forward::primitive_desc(poolDesc, poolAttr, engine->getDNNLEngine());
  }

  size_t DNNLPool::getScratchByteSize() const
  {
    return primDesc.scratchpad_desc().get_size();
  }

  void DNNLPool::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
    args[DNNL_ARG_SCRATCHPAD] = getDNNL(scratch);
  }

  void DNNLPool::updateSrc()
  {
    args[DNNL_ARG_SRC] = getDNNL(src);
  }

  void DNNLPool::updateDst()
  {
    args[DNNL_ARG_DST] = getDNNL(dst);
  }

  void DNNLPool::finalize()
  {
    if (prim)
      throw std::logic_error("pooling already finalized");

    prim = dnnl::pooling_forward(primDesc);
  }

  void DNNLPool::submit()
  {
    if (!prim)
      throw std::logic_error("pooling not finalized");
    if (!src || !dst)
      throw std::logic_error("pooling source/destination not set");

    prim.execute(engine->getDNNLStream(), args);
  }

} // namespace oidn