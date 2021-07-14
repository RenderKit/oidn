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
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Image> output;
    std::shared_ptr<TransferFunction> transferFunc;

    ispc::OutputReorder impl;

  public:
    OutputReorderNode(const Ref<Device>& device,
                      const std::shared_ptr<Tensor>& src,
                      const std::shared_ptr<TransferFunction>& transferFunc,
                      bool hdr,
                      bool snorm)
      : Node(device),
        src(src),
        transferFunc(transferFunc)
    {
      assert(src->ndims() == 3);
      assert(src->layout == TensorLayout::chw ||
             src->layout == TensorLayout::Chw8c ||
             src->layout == TensorLayout::Chw16c);
      assert(src->blockSize() == device->getTensorBlockSize());

      setTile(0, 0, 0, 0, 0, 0);
      impl.transferFunc = transferFunc->getImpl();
      impl.hdr = hdr;
      impl.snorm = snorm;
    }

    void setDst(const std::shared_ptr<Image>& output)
    {
      assert(output);
      assert(src->dims[0] >= output->numChannels());
      assert(output->numChannels() == 3);

      this->output = output;
    }

    void setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
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
      impl.src = *src;
      impl.output = *output;

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
