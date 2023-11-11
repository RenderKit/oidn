// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "output_process.h"

OIDN_NAMESPACE_BEGIN

  OutputProcess::OutputProcess(const OutputProcessDesc& desc)
    : OutputProcessDesc(desc)
  {
    if (srcDesc.getRank() != 3)
      throw std::invalid_argument("invalid output processing source shape");

    setTile(0, 0, 0, 0, 0, 0);
  }

  void OutputProcess::setSrc(const Ref<Tensor>& src)
  {
    if (!src || src->getDesc() != srcDesc)
      throw std::invalid_argument("invalid output processing source");

    this->src = src;
  }

  void OutputProcess::setDst(const Ref<Image>& dst)
  {
    if (!dst || dst->getC() > srcDesc.getC())
      throw std::invalid_argument("invalid output processing destination");

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

  void OutputProcess::check()
  {
    if (!src || !dst)
      throw std::logic_error("output processing source/destination not set");
    if (tile.hSrcBegin + tile.H > src->getH() ||
        tile.wSrcBegin + tile.W > src->getW() ||
        tile.hDstBegin + tile.H > dst->getH() ||
        tile.wDstBegin + tile.W > dst->getW())
      throw std::out_of_range("output processing source/destination out of bounds");
  }

OIDN_NAMESPACE_END