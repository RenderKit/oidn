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
                     const Image& color,
                     const Image& albedo,
                     const Image& normal,
                     const Ref<Tensor>& dst,
                     const Ref<TransferFunction>& transferFunc,
                     bool hdr,
                     bool snorm)
      : Node(device),
        color(color), albedo(albedo), normal(normal),
        dst(dst),
        transferFunc(transferFunc)
    {
      assert(dst->ndims() == 3);
      assert(dst->layout == TensorLayout::chw ||
             dst->layout == TensorLayout::Chw8c ||
             dst->layout == TensorLayout::Chw16c);
      assert(dst->blockSize() == device->getTensorBlockSize());
      assert(dst->dims[0] >= color.numChannels()  +
                             albedo.numChannels() +
                             normal.numChannels());

      impl.color  = color;
      impl.albedo = albedo;
      impl.normal = normal;

      impl.dst = *dst;

      impl.hSrcBegin = 0;
      impl.wSrcBegin = 0;
      impl.hDstBegin = 0;
      impl.wDstBegin = 0;
      impl.H = color.height;
      impl.W = color.width;

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
      assert(impl.H + impl.hSrcBegin <= color.height);
      assert(impl.W + impl.wSrcBegin <= color.width);
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
