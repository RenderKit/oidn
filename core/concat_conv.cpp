// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "concat_conv.h"

namespace oidn {

  ConcatConv::ConcatConv(const ConcatConvDesc& desc)
    : ConcatConvDesc(desc)
  {
    assert(src1Desc.getRank() == 3);
    assert(src2Desc.getRank() == 3);
    assert(src2Desc.getH() == src1Desc.getH() && src2Desc.getW() == src1Desc.getW());
    assert(src2Desc.layout == src1Desc.layout && src2Desc.dataType == src1Desc.dataType);
    assert(weight->getRank() == 4);
    assert(weight->getI() == (src1Desc.getC() + src2Desc.getC()));
    assert(bias->getRank() == 1);
    assert(bias->getX() == weight->getO());

    TensorDims dstDims {weight->getO(), src1Desc.getH(), src1Desc.getW()};
    dstDesc = TensorDesc(dstDims, src1Desc.layout, src1Desc.dataType);
  }

  void ConcatConv::setSrc(const std::shared_ptr<Tensor>& src1, const std::shared_ptr<Tensor>& src2)
  {
    assert(src1->getDesc() == src1Desc);
    assert(src2->getDesc() == src2Desc);
    this->src1 = src1;
    this->src2 = src2;
  }

  void ConcatConv::setDst(const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->getDesc() == dstDesc);
    this->dst = dst;
  }

  CHWConcatConv::CHWConcatConv(const Ref<Device>& device, const ConcatConvDesc& desc)
    : BaseOp(device),
      ConcatConv(desc)
  {
    assert(src1Desc.layout != TensorLayout::hwc);
    TensorDims srcDims {src1Desc.getC() + src2Desc.getC(), src1Desc.getH(), src1Desc.getW()};
    srcDesc = TensorDesc(srcDims, src1Desc.layout, src1Desc.dataType);
    conv = device->newConv({srcDesc, weight, bias, relu});
  }

  void CHWConcatConv::setSrc(const std::shared_ptr<Tensor>& src1, const std::shared_ptr<Tensor>& src2)
  {
    assert(src1->getBuffer() == src2->getBuffer() && ((char*)src1->getData() + src1->getByteSize()) == (char*)src2->getData());
    ConcatConv::setSrc(src1, src2);

    std::shared_ptr<Tensor> src;
    if (src1->getBuffer())
      src = src1->getBuffer()->newTensor(srcDesc, src1->getByteOffset());
    else
      src = device->newTensor(srcDesc, src1->getData());
    conv->setSrc(src);
  }

  void CHWConcatConv::setDst(const std::shared_ptr<Tensor>& dst)
  {
    ConcatConv::setDst(dst);
    conv->setDst(dst);
  }

} // namespace oidn
