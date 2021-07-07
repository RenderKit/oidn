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
  protected:
    std::shared_ptr<Image> color;
    std::shared_ptr<Image> albedo;
    std::shared_ptr<Image> normal;
    std::shared_ptr<Tensor> dst;
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
    InputReorderNode(const Ref<Device>& device,
                     const std::shared_ptr<Tensor>& dst,
                     const std::shared_ptr<TransferFunction>& transferFunc,
                     bool hdr,
                     bool snorm)
      : Node(device),
        dst(dst),
        transferFunc(transferFunc),
        hdr(hdr),
        snorm(snorm)
    {
      assert(dst->ndims() == 3);
      assert(dst->layout == TensorLayout::chw ||
             dst->layout == TensorLayout::Chw8c ||
             dst->layout == TensorLayout::Chw16c);
      assert(dst->blockSize() == device->getTensorBlockSize());
    }

    void setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal)
    {
      assert(dst->dims[0] >= (color  ? color->numChannels()  : 0) +
                             (albedo ? albedo->numChannels() : 0) +
                             (normal ? normal->numChannels() : 0));

      this->color  = color;
      this->albedo = albedo;
      this->normal = normal;
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

    std::shared_ptr<Tensor> getDst() const override { return dst; }

  protected:
    int getWidth() const
    {
      return color ? color->width : (albedo ? albedo->width : normal->width);
    }

    int getHeight() const
    {
      return color ? color->height : (albedo ? albedo->height : normal->height);
    }
  };

  class CPUInputReorderNode : public InputReorderNode
  {
  public:
    CPUInputReorderNode(const Ref<Device>& device,
                        const std::shared_ptr<Tensor>& dst,
                        const std::shared_ptr<TransferFunction>& transferFunc,
                        bool hdr,
                        bool snorm)
      : InputReorderNode(device, dst, transferFunc, hdr, snorm) {}

    void execute() override
    {
      assert(H + hSrcBegin <= getHeight());
      assert(W + wSrcBegin <= getWidth());
      assert(H + hDstBegin <= dst.H);
      assert(W + wDstBegin <= dst.W);

      ispc::InputReorder impl;

      impl.color  = color  ? *color  : Image();
      impl.albedo = albedo ? *albedo : Image();
      impl.normal = normal ? *normal : Image();
      impl.dst = *dst;

      impl.hSrcBegin = hSrcBegin;
      impl.wSrcBegin = wSrcBegin;
      impl.hDstBegin = hDstBegin;
      impl.wDstBegin = wDstBegin;
      impl.H = H;
      impl.W = W;

      impl.transferFunc = *transferFunc;
      impl.hdr = hdr;
      impl.snorm = snorm;

      parallel_nd(impl.dst.H, [&](int hDst)
      {
        ispc::InputReorder_kernel(&impl, hDst);
      });
    }
  };

} // namespace oidn
