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
  protected:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Image> output;
    std::shared_ptr<TransferFunction> transferFunc;

    // Tile
    int hSrcBegin = 0;
    int wSrcBegin = 0;
    int hDstBegin = 0;
    int wDstBegin = 0;
    int H = 0;
    int W = 0;

    bool hdr;
    bool snorm;

  public:
    OutputReorderNode(const Ref<Device>& device,
                      const std::shared_ptr<Tensor>& src,
                      const std::shared_ptr<TransferFunction>& transferFunc,
                      bool hdr,
                      bool snorm)
      : Node(device),
        src(src),
        transferFunc(transferFunc),
        hdr(hdr),
        snorm(snorm)
    {
      assert(src->ndims() == 3);
      assert(src->layout == TensorLayout::chw ||
             src->layout == TensorLayout::Chw8c ||
             src->layout == TensorLayout::Chw16c);
      assert(src->blockSize() == device->getTensorBlockSize());
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
      this->hSrcBegin = hSrc;
      this->wSrcBegin = wSrc;
      this->hDstBegin = hDst;
      this->wDstBegin = wDst;
      this->H = H;
      this->W = W;
    }
  };

  class CPUOutputReorderNode : public OutputReorderNode
  {
  public:
    CPUOutputReorderNode(const Ref<Device>& device,
                         const std::shared_ptr<Tensor>& src,
                         const std::shared_ptr<TransferFunction>& transferFunc,
                         bool hdr,
                         bool snorm)
      : OutputReorderNode(device, src, transferFunc, hdr, snorm) {}

    void execute() override
    {
      assert(hSrcBegin + H <= src->dims[1]);
      assert(wSrcBegin + W <= src->dims[2]);
      //assert(hDstBegin + H <= output->height);
      //assert(wDstBegin + W <= output->width);

      ispc::OutputReorder impl;

      impl.src = *src;
      impl.output = *output;

      impl.hSrcBegin = hSrcBegin;
      impl.wSrcBegin = wSrcBegin;
      impl.hDstBegin = hDstBegin;
      impl.wDstBegin = wDstBegin;
      impl.H = H;
      impl.W = W;

      impl.transferFunc = *transferFunc;
      impl.hdr = hdr;
      impl.snorm = snorm;

      parallel_nd(impl.H, [&](int h)
      {
        ispc::OutputReorder_kernel(&impl, h);
      });
    }
  };

} // namespace oidn
