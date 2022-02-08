// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_op.h"

namespace oidn {

  DNNLOp::DNNLOp(const Ref<DNNLDevice>& device)
    : BaseOp(device) {}

  size_t DNNLOp::getScratchSize() const
  {
    const auto primDesc = prim.get_primitive_desc();
    const dnnl_memory_desc_t* scratchpadDesc = dnnl_primitive_desc_query_md(primDesc, dnnl_query_scratchpad_md, 0);
    if (scratchpadDesc == nullptr)
      return 0;
    return dnnl_memory_desc_get_size(scratchpadDesc);
  }

  void DNNLOp::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
    args.insert(std::make_pair(DNNL_ARG_SCRATCHPAD, getDNNL(*scratch)));
  }

  void DNNLOp::run()
  {
    prim.execute(device->getDNNLStream(), args);
  }

} // namespace oidn
