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
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    UpsampleNode(int K,
                 const std::shared_ptr<memory>& src,
                 const std::shared_ptr<memory>& dst)
      : K(K),
        src(src),
        dst(dst)
    {
      data.src = toIspc(src);
      data.dst = toIspc(dst);

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

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

} // namespace oidn
