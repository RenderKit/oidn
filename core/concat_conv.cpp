// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "concat_conv.h"

namespace oidn {

  ConcatConv::ConcatConv(const ConcatConvDesc& desc)
    : ConcatConvDesc(desc)
  {
    if (src1Desc.getRank() != 3 ||
        src2Desc.getRank() != 3 ||
        src1Desc.getH() != src2Desc.getH() ||
        src1Desc.getW() != src2Desc.getW() ||
        src1Desc.layout != src2Desc.layout ||
        src1Desc.dataType != src2Desc.dataType)
      throw std::invalid_argument("invalid concatenation+convolution source descriptor");
    if (weightDesc.getRank() != 4 || weightDesc.getI() != (src1Desc.getC() + src2Desc.getC()))
      throw std::invalid_argument("invalid concatenation+convolution weight shape");
    if (biasDesc.getRank() != 1 || biasDesc.getX() != weightDesc.getO())
      throw std::invalid_argument("invalid concatenation+convolution bias shape");

    TensorDims dstDims {weightDesc.getO(), src1Desc.getH(), src1Desc.getW()};
    dstDesc = TensorDesc(dstDims, src1Desc.layout, src1Desc.dataType);
  }

  void ConcatConv::setSrc(const std::shared_ptr<Tensor>& src1, const std::shared_ptr<Tensor>& src2)
  {
    if (!src1 || src1->getDesc() != src1Desc || !src2 || src2->getDesc() != src2Desc)
      throw std::invalid_argument("invalid concatenation+convolution source");

    this->src1 = src1;
    this->src2 = src2;
    updateSrc();
  }

  void ConcatConv::setWeight(const std::shared_ptr<Tensor>& weight)
  {
    if (!weight || weight->getDesc() != weightDesc)
      throw std::invalid_argument("invalid concatenation+convolution weight");

    this->weight = weight;
    updateWeight();
  }
  
  void ConcatConv::setBias(const std::shared_ptr<Tensor>& bias)
  {
    if (!bias || bias->getDesc() != biasDesc)
      throw std::invalid_argument("invalid concatenation+convolution bias");

    this->bias = bias;
    updateBias();
  }

  void ConcatConv::setDst(const std::shared_ptr<Tensor>& dst)
  {
    if (!dst || dst->getDesc() != dstDesc)
      throw std::invalid_argument("invalid concatenation+convolution destination");

    this->dst = dst;
    updateDst();
  }

  CHWConcatConv::CHWConcatConv(const Ref<Engine>& engine, const ConcatConvDesc& desc)
    : ConcatConv(desc),
      engine(engine)
  {
    if (src1Desc.layout == TensorLayout::hwc)
      throw std::invalid_argument("unsupported concatenation+convolution source layout");

    TensorDims srcDims {src1Desc.getC() + src2Desc.getC(), src1Desc.getH(), src1Desc.getW()};
    srcDesc = TensorDesc(srcDims, src1Desc.layout, src1Desc.dataType);
    conv = engine->newConv({srcDesc, weightDesc, biasDesc, activation});
  }

  void CHWConcatConv::updateSrc()
  {
    if (src1->getBuffer() != src2->getBuffer() ||
        ((char*)src1->getData() + src1->getByteSize()) != (char*)src2->getData())
      throw std::invalid_argument("unsupported concatenation+convolution source");

    std::shared_ptr<Tensor> src;
    if (src1->getBuffer())
      src = src1->getBuffer()->newTensor(srcDesc, src1->getByteOffset());
    else
      src = engine->newTensor(srcDesc, src1->getData());
    conv->setSrc(src);
  }

} // namespace oidn
