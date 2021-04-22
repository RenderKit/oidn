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

    Image color;
    Image albedo;
    Image normal;
    Ref<Tensor> dst;
    Ref<TransferFunction> transferFunc;

  public:
    InputReorderNode(const Ref<Device>& device,
                     const Ref<Tensor>& dst,
                     const Ref<TransferFunction>& transferFunc,
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

      impl.dst = *dst;
      setTile(0, 0, 0, 0, 0, 0);
      impl.transferFunc = transferFunc->getImpl();
      impl.hdr = hdr;
      impl.snorm = snorm;
    }

    void setSrc(const Image& color, const Image& albedo, const Image& normal)
    {
      assert(dst->dims[0] >= color.numChannels()  +
                             albedo.numChannels() +
                             normal.numChannels());

      this->color  = color;
      this->albedo = albedo;
      this->normal = normal;

      impl.color  = color;
      impl.albedo = albedo;
      impl.normal = normal;
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
      assert(impl.H + impl.hSrcBegin <= getHeight());
      assert(impl.W + impl.wSrcBegin <= getWidth());
      assert(impl.H + impl.hDstBegin <= impl.dst.H);
      assert(impl.W + impl.wDstBegin <= impl.dst.W);

      parallel_nd(impl.dst.H, [&](int hDst)
      {
        ispc::InputReorder_kernel(&impl, hDst);
      });
    }

    Ref<Tensor> getDst() const override { return dst; }

  private:
    int getWidth() const
    {
      return color ? color.width : (albedo ? albedo.width : normal.width);
    }

    int getHeight() const
    {
      return color ? color.height : (albedo ? albedo.height : normal.height);
    }
  };

} // namespace oidn
