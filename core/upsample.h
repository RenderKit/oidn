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
    ispc::Upsample impl;

    int K;
    Ref<Tensor> src;
    Ref<Tensor> dst;

  public:
    UpsampleNode(const Ref<Device>& device,
                 int K,
                 const Ref<Tensor>& src,
                 const Ref<Tensor>& dst)
      : Node(device),
        K(K),
        src(src),
        dst(dst)
    {
      impl.src = *src;
      impl.dst = *dst;

      assert(impl.dst.H == impl.src.H * 2);
      assert(impl.dst.W == impl.src.W * 2);
    }

    void execute() override
    {
      parallel_nd(impl.src.C / K, impl.src.H, [&](int ck, int h)
      {
        ispc::Upsample_kernel(&impl, ck, h);
      });
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
