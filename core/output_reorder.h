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
    Image output;
    Ref<TransferFunction> transferFunc;

  public:
    OutputReorderNode(const Ref<Device>& device,
                      const Ref<Tensor>& src,
                      const Image& output,
                      const Ref<TransferFunction>& transferFunc,
                      bool hdr,
                      bool snorm)
      : Node(device),
        src(src),
        output(output),
        transferFunc(transferFunc)
    {
      assert(src->ndims() == 3);
      assert(src->layout == TensorLayout::chw ||
             src->layout == TensorLayout::Chw8c ||
             src->layout == TensorLayout::Chw16c);
      assert(src->blockSize() == device->getTensorBlockSize());
      assert(src->dims[0] >= output.numChannels());
      assert(output.numChannels() == 3);

      impl.src = *src;
      impl.output = output;

      impl.hSrcBegin = 0;
      impl.wSrcBegin = 0;
      impl.hDstBegin = 0;
      impl.wDstBegin = 0;
      impl.H = output.height;
      impl.W = output.width;

      impl.transferFunc = transferFunc->getImpl();
      impl.hdr = hdr;
      impl.snorm = snorm;
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
      //assert(impl.hDstBegin + impl.H <= impl.output.H);
      //assert(impl.wDstBegin + impl.W <= impl.output.W);

      parallel_nd(impl.H, [&](int h)
      {
        ispc::OutputReorder_kernel(&impl, h);
      });
    }
  };

} // namespace oidn
