// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "output_reorder.h"
#include "output_reorder_ispc.h"

namespace oidn {
  
  OutputReorderNode::OutputReorderNode(const Ref<Device>& device,
                                       const std::string& name,
                                       const std::shared_ptr<Tensor>& src,
                                       const std::shared_ptr<TransferFunction>& transferFunc,
                                       bool hdr,
                                       bool snorm)
    : Node(device, name),
      src(src),
      transferFunc(transferFunc),
      hdr(hdr),
      snorm(snorm)
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

  CPUOutputReorderNode::CPUOutputReorderNode(const Ref<Device>& device,
                                             const std::string& name,
                                             const std::shared_ptr<Tensor>& src,
                                             const std::shared_ptr<TransferFunction>& transferFunc,
                                             bool hdr,
                                             bool snorm)
    : OutputReorderNode(device, name, src, transferFunc, hdr, snorm) {}

  void CPUOutputReorderNode::execute()
  {
    assert(tile.hSrcBegin + tile.H <= src->dims[1]);
    assert(tile.wSrcBegin + tile.W <= src->dims[2]);
    //assert(tile.hDstBegin + tile.H <= output->height);
    //assert(tile.wDstBegin + tile.W <= output->width);

    ispc::OutputReorder impl;

    impl.src = *src;
    impl.output = *output;
    impl.tile = tile;
    impl.transferFunc = *transferFunc;
    impl.hdr = hdr;
    impl.snorm = snorm;

    parallel_nd(impl.tile.H, [&](int h)
    {
      ispc::OutputReorder_kernel(&impl, h);
    });
  }

} // namespace oidn