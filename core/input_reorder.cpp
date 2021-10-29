// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_reorder.h"
#include "input_reorder_ispc.h"

namespace oidn {

  InputReorderNode::InputReorderNode(const Ref<Device>& device,
                                     const std::string& name,
                                     const std::shared_ptr<Tensor>& dst,
                                     const std::shared_ptr<TransferFunction>& transferFunc,
                                     bool hdr,
                                     bool snorm)
    : Node(device, name),
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

    setTile(0, 0, 0, 0, 0, 0);
  }

  void InputReorderNode::setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal)
  {
    assert(dst->dims[0] >= (color  ? color->numChannels()  : 0) +
                           (albedo ? albedo->numChannels() : 0) +
                           (normal ? normal->numChannels() : 0));

    this->color  = color;
    this->albedo = albedo;
    this->normal = normal;
  }

  void InputReorderNode::setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
  {
    tile.hSrcBegin = hSrc;
    tile.wSrcBegin = wSrc;
    tile.hDstBegin = hDst;
    tile.wDstBegin = wDst;
    tile.H = H;
    tile.W = W;
  }

  CPUInputReorderNode::CPUInputReorderNode(const Ref<Device>& device,
                                           const std::string& name,
                                           const std::shared_ptr<Tensor>& dst,
                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                           bool hdr,
                                           bool snorm)
    : InputReorderNode(device, name, dst, transferFunc, hdr, snorm) {}

  void CPUInputReorderNode::execute()
  {
    assert(tile.H + tile.hSrcBegin <= getInput()->height);
    assert(tile.W + tile.wSrcBegin <= getInput()->width);
    assert(tile.H + tile.hDstBegin <= dst->height());
    assert(tile.W + tile.wDstBegin <= dst->width());

    ispc::InputReorder impl;

    impl.color  = color  ? *color  : Image();
    impl.albedo = albedo ? *albedo : Image();
    impl.normal = normal ? *normal : Image();
    impl.dst = *dst;
    impl.tile = tile;
    impl.transferFunc = *transferFunc;
    impl.hdr = hdr;
    impl.snorm = snorm;

    parallel_nd(impl.dst.H, [&](int hDst)
    {
      ispc::InputReorder_kernel(&impl, hDst);
    });
  }

} // namespace oidn