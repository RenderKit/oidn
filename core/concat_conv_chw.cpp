// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "concat_conv_chw.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  ConcatConvCHW::ConcatConvCHW(Engine* engine, const ConcatConvDesc& desc)
    : ConcatConv(desc)
  {
    if (src1Desc.layout == TensorLayout::hwc)
      throw std::invalid_argument("unsupported concat+conv source layout");

    TensorDims srcDims{src1Desc.getC() + src2Desc.getC(), src1Desc.getH(), src1Desc.getW()};
    TensorDims srcPaddedDims{src1Desc.getPaddedC() + src2Desc.getPaddedC(), src1Desc.getH(), src1Desc.getW()};
    srcDesc = {srcDims, srcPaddedDims, src1Desc.layout, src1Desc.dataType};

    conv = engine->newConv({srcDesc, weightDesc, biasDesc, activation, PostOp::None, fastMath});
  }

  void ConcatConvCHW::updateSrc()
  {
    if (!src1->getBuffer() || !src2->getBuffer())
      throw std::invalid_argument("concat+conv sources must be backed by buffers");
    if (src1->getBuffer() != src2->getBuffer() ||
        (static_cast<char*>(src1->getPtr()) + src1->getByteSize()) != static_cast<char*>(src2->getPtr()))
      throw std::invalid_argument("concat+conv sources are not pre-concatenated in memory");

    auto src = src1->getBuffer()->newTensor(srcDesc, src1->getByteOffset());
    conv->setSrc(src);
  }

OIDN_NAMESPACE_END