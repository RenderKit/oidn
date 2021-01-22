// Copyright 2009-2021 Intel Corporation
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
    ispc::InputReorder impl;

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
      assert(srcColor);
      assert(dst->ndims() == 3);
      assert(dst->layout == TensorLayout::chw ||
             dst->layout == TensorLayout::Chw8c ||
             dst->layout == TensorLayout::Chw16c);
      assert(dst->blockSize() == device->getTensorBlockSize());
      assert(dst->dims[0] >= srcColor.numChannels()  +
                             srcAlbedo.numChannels() +
                             srcNormal.numChannels());

      impl.srcColor  = srcColor;
      impl.srcAlbedo = srcAlbedo;
      impl.srcNormal = srcNormal;

      impl.dst = *dst;

      impl.hSrcBegin = 0;
      impl.wSrcBegin = 0;
      impl.hDstBegin = 0;
      impl.wDstBegin = 0;
      impl.H = srcColor.height;
      impl.W = srcColor.width;

      impl.transferFunc = transferFunc->getImpl();
      impl.hdr = hdr;
    }

    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W) override
    {
      impl.hSrcBegin = hSrc;
      impl.wSrcBegin = wSrc;
      impl.hDstBegin = hDst;
      impl.wDstBegin = wDst;
      impl.H = H;
      impl.W = W;
    }

    void execute() override
    {
      assert(impl.H + impl.hSrcBegin <= srcColor.height);
      assert(impl.W + impl.wSrcBegin <= srcColor.width);
      assert(impl.H + impl.hDstBegin <= impl.dst.H);
      assert(impl.W + impl.wDstBegin <= impl.dst.W);

      parallel_nd(impl.dst.H, [&](int hDst)
      {
        ispc::InputReorder_kernel(&impl, hDst);
      });
    }

    Ref<Tensor> getDst() const override { return dst; }
  };

} // namespace oidn
