// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../reorder.h"
#include "dnnl_common.h"

namespace oidn {

  class DNNLReorder final : public Op
  {
  public:
    DNNLReorder(const Ref<DNNLEngine>& engine, const ReorderDesc& desc)
      : engine(engine),
        src(desc.src),
        dst(desc.dst)
    {
      const dnnl::memory& srcMem = getDNNL(src);
      const dnnl::memory& dstMem = getDNNL(dst);

      prim = dnnl::reorder(dnnl::reorder::primitive_desc(srcMem, dstMem));
      args = {{DNNL_ARG_SRC, srcMem},
              {DNNL_ARG_DST, dstMem}};
    }

    void submit() override
    {
      prim.execute(engine->getDNNLStream(), args);
    }
  
  private:
    Ref<DNNLEngine> engine;
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
    dnnl::reorder prim;
    std::unordered_map<int, dnnl::memory> args;
  };

} // namespace oidn
