// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"
#include "image.h"
#include "color.h"
#include "output_reorder_ispc.h"

namespace oidn {

  // Output reorder node
  class OutputReorderNode : public Node
  {
  private:
    ispc::OutputReorder data;

    std::shared_ptr<memory> src;
    Image dst;
    std::shared_ptr<TransferFunction> transferFunc;

  public:
    OutputReorderNode(const std::shared_ptr<memory>& src,
                      const Image& dst,
                      const std::shared_ptr<TransferFunction>& transferFunc,
                      bool hdr)
      : src(src),
        dst(dst),
        transferFunc(transferFunc)
    {
      data.src = toIspc(src);
      data.dst = toIspc(dst);

      data.hSrcBegin = 0;
      data.wSrcBegin = 0;
      data.hDstBegin = 0;
      data.wDstBegin = 0;
      data.H = dst.height;
      data.W = dst.width;

      data.transferFunc = transferFunc->getIspc();
      data.hdr = hdr;
    }

    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W) override
    {
      data.hSrcBegin = hSrc;
      data.wSrcBegin = wSrc;
      data.hDstBegin = hDst;
      data.wDstBegin = wDst;
      data.H = H;
      data.W = W;
    }

    void execute(stream& sm) override
    {
      assert(data.hSrcBegin + data.H <= data.src.H);
      assert(data.wSrcBegin + data.W <= data.src.W);
      //assert(data.hDstBegin + data.H <= data.dst.H);
      //assert(data.wDstBegin + data.W <= data.dst.W);

      parallel_nd(data.H, [&](int h)
      {
        ispc::OutputReorder_kernel(&data, h);
      });
    }
  };

} // namespace oidn
