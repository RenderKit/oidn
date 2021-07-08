// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#if defined(OIDN_DEVICE_GPU)
  #include "sycl_device.h"
#endif

#include "output_reorder.h"
#include "output_reorder_ispc.h"

namespace oidn {
  
  OutputReorderNode::OutputReorderNode(const Ref<Device>& device,
                                       const std::shared_ptr<Tensor>& src,
                                       const std::shared_ptr<TransferFunction>& transferFunc,
                                       bool hdr,
                                       bool snorm)
    : Node(device),
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
    this->hSrcBegin = hSrc;
    this->wSrcBegin = wSrc;
    this->hDstBegin = hDst;
    this->wDstBegin = wDst;
    this->H = H;
    this->W = W;
  }

  CPUOutputReorderNode::CPUOutputReorderNode(const Ref<Device>& device,
                                             const std::shared_ptr<Tensor>& src,
                                             const std::shared_ptr<TransferFunction>& transferFunc,
                                             bool hdr,
                                             bool snorm)
    : OutputReorderNode(device, src, transferFunc, hdr, snorm) {}

  void CPUOutputReorderNode::execute()
  {
    assert(hSrcBegin + H <= src->dims[1]);
    assert(wSrcBegin + W <= src->dims[2]);
    //assert(hDstBegin + H <= output->height);
    //assert(wDstBegin + W <= output->width);

    ispc::OutputReorder impl;

    impl.src = *src;
    impl.output = *output;

    impl.hSrcBegin = hSrcBegin;
    impl.wSrcBegin = wSrcBegin;
    impl.hDstBegin = hDstBegin;
    impl.wDstBegin = wDstBegin;
    impl.H = H;
    impl.W = W;

    impl.transferFunc = *transferFunc;
    impl.hdr = hdr;
    impl.snorm = snorm;

    parallel_nd(impl.H, [&](int h)
    {
      ispc::OutputReorder_kernel(&impl, h);
    });
  }

#if defined(OIDN_DEVICE_GPU)

  SYCLOutputReorderNode::SYCLOutputReorderNode(const Ref<SYCLDevice>& device,
                                               const std::shared_ptr<Tensor>& src,
                                               const std::shared_ptr<TransferFunction>& transferFunc,
                                               bool hdr,
                                               bool snorm)
    : OutputReorderNode(device, src, transferFunc, hdr, snorm) {}

  void SYCLOutputReorderNode::execute()
  {
    assert(hSrcBegin + H <= src->dims[1]);
    assert(wSrcBegin + W <= src->dims[2]);
    //assert(hDstBegin + H <= output->height);
    //assert(wDstBegin + W <= output->width);
  }

#endif

} // namespace oidn