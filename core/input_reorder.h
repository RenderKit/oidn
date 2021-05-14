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
    std::shared_ptr<Image> color;
    std::shared_ptr<Image> albedo;
    std::shared_ptr<Image> normal;
    std::shared_ptr<Tensor> dst;
    std::shared_ptr<TransferFunction> transferFunc;

    ispc::InputReorder impl;

  public:
    InputReorderNode(const Ref<Device>& device,
                     const std::shared_ptr<Tensor>& dst,
                     const std::shared_ptr<TransferFunction>& transferFunc,
                     bool hdr,
                     bool snorm)
      : Node(device),
        dst(dst),
        transferFunc(transferFunc)
    {
      assert(dst->ndims() == 3);
      assert(dst->layout == TensorLayout::chw ||
             dst->layout == TensorLayout::Chw8c ||
             dst->layout == TensorLayout::Chw16c);
      assert(dst->blockSize() == device->getTensorBlockSize());

      setTile(0, 0, 0, 0, 0, 0);
      impl.transferFunc = transferFunc->getImpl();
      impl.hdr = hdr;
      impl.snorm = snorm;
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
      impl.hSrcBegin = hSrc;
      impl.wSrcBegin = wSrc;
      impl.hDstBegin = hDst;
      impl.wDstBegin = wDst;
      impl.H = H;
      impl.W = W;
    }

    void execute() override
    {
      impl.color  = color  ? *color  : Image();
      impl.albedo = albedo ? *albedo : Image();
      impl.normal = normal ? *normal : Image();
      impl.dst = *dst;

      assert(impl.H + impl.hSrcBegin <= getHeight());
      assert(impl.W + impl.wSrcBegin <= getWidth());
      assert(impl.H + impl.hDstBegin <= impl.dst.H);
      assert(impl.W + impl.wDstBegin <= impl.dst.W);

      parallel_nd(impl.dst.H, [&](int hDst)
      {
        ispc::InputReorder_kernel(&impl, hDst);
      });
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }

  private:
    int getWidth() const
    {
      return color ? color->width : (albedo ? albedo->width : normal->width);
    }

    int getHeight() const
    {
      return color ? color->height : (albedo ? albedo->height : normal->height);
    }
  };

} // namespace oidn
