// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "output_reorder.h"

namespace oidn {
  
  OutputReorderNode::OutputReorderNode(const OutputReorderDesc& desc)
    : src(desc.src),
      transferFunc(desc.transferFunc),
      hdr(desc.hdr),
      snorm(desc.snorm)
  {
    assert(src->ndims() == 3);
    assert(src->layout == TensorLayout::chw ||
           src->layout == TensorLayout::Chw8c ||
           src->layout == TensorLayout::Chw16c);
    assert(src->blockSize() == device->getTensorBlockSize());

    setTile(0, 0, 0, 0, 0, 0);
  }

  void OutputReorderNode::setDst(const std::shared_ptr<Image>& output)
  {
    assert(output);
    assert(src->dims[0] >= output->numChannels());
    assert(output->numChannels() == 3);

    this->output = output;
  }

  void OutputReorderNode::setTile(int hSrc, int wSrc, int hDst, int wDst, int H, int W)
  {
    tile.hSrcBegin = hSrc;
    tile.wSrcBegin = wSrc;
    tile.hDstBegin = hDst;
    tile.wDstBegin = wDst;
    tile.H = H;
    tile.W = W;
  }

} // namespace oidn