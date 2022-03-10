// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "output_process.h"

namespace oidn {
  
  OutputProcess::OutputProcess(const OutputProcessDesc& desc)
    : OutputProcessDesc(desc)
  {
    assert(srcDesc.getRank() == 3);
    setTile(0, 0, 0, 0, 0, 0);
  }

  void OutputProcess::setSrc(const std::shared_ptr<Tensor>& src)
  {
    assert(src->getDesc() == srcDesc);
    this->src = src;
  }

  void OutputProcess::setDst(const std::shared_ptr<Image>& dst)
  {
    assert(srcDesc.getC() >= dst->getC());
    assert(dst->getC() == 3);
    this->dst = dst;
  }

  void OutputProcess::setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
  {
    tile.hSrcBegin = hSrc;
    tile.wSrcBegin = wSrc;
    tile.hDstBegin = hDst;
    tile.wDstBegin = wDst;
    tile.H = H;
    tile.W = W;
  }

} // namespace oidn