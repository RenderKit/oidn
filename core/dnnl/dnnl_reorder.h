// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../reorder.h"
#include "dnnl_node.h"

namespace oidn {

  // Reorder node
  class DNNLReorderNode : public DNNLNode
  {
  private:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    DNNLReorderNode(const Ref<DNNLDevice>& device, const ReorderDesc& desc)
      : DNNLNode(device, desc.name),
        src(desc.src),
        dst(desc.dst)
    {
      const dnnl::memory& srcMem = DNNLTensor::getMemory(*src);
      const dnnl::memory& dstMem = DNNLTensor::getMemory(*dst);

      prim = dnnl::reorder(dnnl::reorder::primitive_desc(srcMem, dstMem));
      args = {{DNNL_ARG_SRC, srcMem},
              {DNNL_ARG_DST, dstMem}};
    }

    //std::shared_ptr<Tensor> getDst() const { return dst; }
  };

} // namespace oidn
