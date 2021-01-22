// Copyright 2009-2021 Intel Corporation
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
    ispc::OutputReorder impl;

    Ref<Tensor> src;
    Image dst;
    Ref<TransferFunction> transferFunc;

  public:
    OutputReorderNode(const Ref<Device>& device,
                      const Ref<Tensor>& src,
                      const Image& dst,
                      const Ref<TransferFunction>& transferFunc,
                      bool hdr)
      : Node(device),
        src(src),
        dst(dst),
        transferFunc(transferFunc)
    {
      assert(src->ndims() == 3);
      assert(src->layout == TensorLayout::chw ||
             src->layout == TensorLayout::Chw8c ||
             src->layout == TensorLayout::Chw16c);
      assert(src->blockSize() == device->getTensorBlockSize());
      assert(src->dims[0] >= dst.numChannels());
      assert(dst.numChannels() == 3);

      impl.src = *src;
      impl.dst = dst;

      impl.hSrcBegin = 0;
      impl.wSrcBegin = 0;
      impl.hDstBegin = 0;
      impl.wDstBegin = 0;
      impl.H = dst.height;
      impl.W = dst.width;

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
      assert(impl.hSrcBegin + impl.H <= impl.src.H);
      assert(impl.wSrcBegin + impl.W <= impl.src.W);
      //assert(impl.hDstBegin + impl.H <= impl.dst.H);
      //assert(impl.wDstBegin + impl.W <= impl.dst.W);

      parallel_nd(impl.H, [&](int h)
      {
        ispc::OutputReorder_kernel(&impl, h);
      });
    }
  };

} // namespace oidn
