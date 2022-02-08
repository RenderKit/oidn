// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "output_process.h"

namespace oidn {
  
  OutputProcess::OutputProcess(const OutputProcessDesc& desc)
    : src(desc.src),
      transferFunc(desc.transferFunc),
      hdr(desc.hdr),
      snorm(desc.snorm)
  {
    assert(src->getRank() == 3);
    assert(src->getBlockSize() == device->getTensorBlockSize());

    setTile(0, 0, 0, 0, 0, 0);
  }

  void OutputProcess::setDst(const std::shared_ptr<Image>& output)
  {
    assert(output);
    assert(src->getC() >= output->getC());
    assert(output->getC() == 3);

    this->output = output;
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