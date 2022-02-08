// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_process.h"

namespace oidn {

  InputProcess::InputProcess(const InputProcessDesc& desc)
    : dst(desc.dst),
      transferFunc(desc.transferFunc),
      hdr(desc.hdr),
      snorm(desc.snorm)
  {
    assert(dst->getRank() == 3);
    assert(dst->getBlockSize() == device->getTensorBlockSize());

    setTile(0, 0, 0, 0, 0, 0);
  }

  void InputProcess::setSrc(const std::shared_ptr<Image>& color, const std::shared_ptr<Image>& albedo, const std::shared_ptr<Image>& normal)
  {
    assert(dst->getC() >= (color  ? color->getC()  : 0) +
                          (albedo ? albedo->getC() : 0) +
                          (normal ? normal->getC() : 0));

    this->color  = color;
    this->albedo = albedo;
    this->normal = normal;
  }

  void InputProcess::setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
  {
    tile.hSrcBegin = hSrc;
    tile.wSrcBegin = wSrc;
    tile.hDstBegin = hDst;
    tile.wDstBegin = wDst;
    tile.H = H;
    tile.W = W;
  }

} // namespace oidn