// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "concat_conv_hwc.h"
#include "engine.h"

OIDN_NAMESPACE_BEGIN

  ConcatConvHWC::ConcatConvHWC(const Ref<Engine>& engine, const ConcatConvDesc& desc)
    : ConcatConv(desc),
      engine(engine)
  {
    if (src1Desc.layout != TensorLayout::hwc)
      throw std::logic_error("unsupported concat+conv source layout");

    // Split the convolution into two smaller convolutions
    weight1Desc = {{dstDesc.getC(),       src1Desc.getC(),       weightDesc.getH(), weightDesc.getW()},
                   {dstDesc.getPaddedC(), src1Desc.getPaddedC(), weightDesc.getH(), weightDesc.getW()},
                   weightDesc.layout,
                   weightDesc.dataType};

    weight2Desc = {{dstDesc.getC(),       src2Desc.getC(),       weightDesc.getH(), weightDesc.getW()},
                   {dstDesc.getPaddedC(), src2Desc.getPaddedC(), weightDesc.getH(), weightDesc.getW()},
                   weightDesc.layout,
                   weightDesc.dataType};

    // Convolution 1: dst = conv(src1, weight1) + bias
    conv1 = engine->newConv({src1Desc, weight1Desc, biasDesc, Activation::None, PostOp::None, fastMath});

    // Convolution 2: dst = activation(conv(src2, weight2) + dst)
    // We use dst as bias
    conv2 = engine->newConv({src2Desc, weight2Desc, dstDesc, activation, PostOp::None, fastMath});
  }

  bool ConcatConvHWC::isSupported() const
  {
    return conv1->isSupported() && conv2->isSupported();
  }

  size_t ConcatConvHWC::getScratchByteSize() const
  {
    return max(conv1->getScratchByteSize(), conv2->getScratchByteSize());
  }

  void ConcatConvHWC::setScratch(const Ref<Buffer>& scratch)
  {
    conv1->setScratch(scratch);
    conv2->setScratch(scratch);
  }

  void ConcatConvHWC::setWeight(const std::shared_ptr<Tensor>& weight1, const std::shared_ptr<Tensor>& weight2)
  {
    conv1->setWeight(weight1);
    conv2->setWeight(weight2);
  }

  void ConcatConvHWC::updateSrc()
  {
    conv1->setSrc(src1);
    conv2->setSrc(src2);
  }

  void ConcatConvHWC::updateDst()
  {
    conv1->setDst(dst);

    conv2->setBias(dst);
    conv2->setDst(dst);
  }

  void ConcatConvHWC::finalize()
  {
    conv1->finalize();
    conv2->finalize();
  }

  void ConcatConvHWC::submit()
  {
    conv1->submit();
    conv2->submit();
  }

OIDN_NAMESPACE_END