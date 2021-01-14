// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"
#include "image.h"
#include "color.h"
#include "input_reorder_ispc.h"

namespace oidn {

  // Input reorder node
  class InputReorderNode : public Node
  {
  private:
    ispc::InputReorder data;

    Image srcColor;
    Image srcAlbedo;
    Image srcNormal;
    Ref<Tensor> dst;
    Ref<TransferFunction> transferFunc;

  public:
    InputReorderNode(const Ref<Device>& device,
                     const Image& srcColor,
                     const Image& srcAlbedo,
                     const Image& srcNormal,
                     const Ref<Tensor>& dst,
                     const Ref<TransferFunction>& transferFunc,
                     bool hdr)
      : Node(device),
        srcColor(srcColor), srcAlbedo(srcAlbedo), srcNormal(srcNormal),
        dst(dst),
        transferFunc(transferFunc)
    {
      data.srcColor  = srcColor;
      data.srcAlbedo = srcAlbedo;
      data.srcNormal = srcNormal;

      data.dst = *dst;

      data.hSrcBegin = 0;
      data.wSrcBegin = 0;
      data.hDstBegin = 0;
      data.wDstBegin = 0;
      data.H = srcColor.height;
      data.W = srcColor.width;

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

    void execute() override
    {
      assert(data.H + data.hSrcBegin <= srcColor.height);
      assert(data.W + data.wSrcBegin <= srcColor.width);
      assert(data.H + data.hDstBegin <= data.dst.H);
      assert(data.W + data.wDstBegin <= data.dst.W);

      parallel_nd(data.dst.H, [&](int hDst)
      {
        ispc::InputReorder_kernel(&data, hDst);
      });
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
