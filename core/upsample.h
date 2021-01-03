// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"
#include "upsample_ispc.h"

namespace oidn {

  // 2x2 nearest-neighbor upsampling node
  class UpsampleNode : public Node
  {
  private:
    ispc::Upsample data;

    int K;
    Ref<Tensor> src;
    Ref<Tensor> dst;

  public:
    UpsampleNode(int K,
                 const Ref<Tensor>& src,
                 const Ref<Tensor>& dst)
      : K(K),
        src(src),
        dst(dst)
    {
      data.src = *src;
      data.dst = *dst;

      assert(data.dst.H == data.src.H * 2);
      assert(data.dst.W == data.src.W * 2);
    }

    void execute(stream& sm) override
    {
      parallel_nd(data.src.C / K, data.src.H, [&](int ck, int h)
      {
        ispc::Upsample_kernel(&data, ck, h);
      });
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
