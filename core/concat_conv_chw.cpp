// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "concat_conv_chw.h"

OIDN_NAMESPACE_BEGIN

  ConcatConvCHW::ConcatConvCHW(const Ref<Engine>& engine, const ConcatConvDesc& desc)
    : ConcatConv(desc),
      engine(engine)
  {
    if (src1Desc.layout == TensorLayout::hwc)
      throw std::invalid_argument("unsupported concatenation+convolution source layout");

    TensorDims srcDims {src1Desc.getC() + src2Desc.getC(), src1Desc.getH(), src1Desc.getW()};
    srcDesc = TensorDesc(srcDims, src1Desc.layout, src1Desc.dataType);
    conv = engine->newConv({srcDesc, weightDesc, biasDesc, activation, PostOp::None});
  }

  void ConcatConvCHW::updateSrc()
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

OIDN_NAMESPACE_END