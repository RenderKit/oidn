// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_node.h"

namespace oidn {

  DNNLNode::DNNLNode(const Ref<DNNLDevice>& device, const std::string& name)
    : BaseNode(device, name) {}

  size_t DNNLNode::getScratchSize() const
  {
    const auto primDesc = prim.get_primitive_desc();
    const dnnl_memory_desc_t* scratchpadDesc = dnnl_primitive_desc_query_md(primDesc, dnnl_query_scratchpad_md, 0);
    if (scratchpadDesc == nullptr)
      return 0;
    return dnnl_memory_desc_get_size(scratchpadDesc);
  }

  void DNNLNode::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    this->scratch = scratch;
    args.insert(std::make_pair(DNNL_ARG_SCRATCHPAD, DNNLTensor::getMemory(*scratch)));
  }

  void DNNLNode::execute()
  {
    prim.execute(device->getDNNLStream(), args);
  }

} // namespace oidn
