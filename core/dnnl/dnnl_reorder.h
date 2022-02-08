// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../reorder.h"
#include "dnnl_op.h"

namespace oidn {

  class DNNLReorder : public DNNLOp
  {
  private:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    DNNLReorder(const Ref<DNNLDevice>& device, const ReorderDesc& desc)
      : DNNLOp(device),
        src(desc.src),
        dst(desc.dst)
    {
      const dnnl::memory& srcMem = getDNNL(*src);
      const dnnl::memory& dstMem = getDNNL(*dst);

      prim = dnnl::reorder(dnnl::reorder::primitive_desc(srcMem, dstMem));
      args = {{DNNL_ARG_SRC, srcMem},
              {DNNL_ARG_DST, dstMem}};
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
