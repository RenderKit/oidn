// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_concat_conv.h"

namespace oidn {

  CUDAConcatConv::CUDAConcatConv(const Ref<CUDAEngine>& engine, const ConcatConvDesc& desc)
    : ConcatConv(desc),
      engine(engine)
  {
    // Split the convolution into two smaller convolutions
    weight1Desc = {{weightDesc.getO(), src1Desc.getC(), weightDesc.getH(), weightDesc.getW()}, weightDesc.layout, weightDesc.dataType};
    weight2Desc = {{weightDesc.getO(), src2Desc.getC(), weightDesc.getH(), weightDesc.getW()}, weightDesc.layout, weightDesc.dataType};

    // Convolution 1: dst = conv(src1, weight1) + bias
    conv1 = newCUDAConv(engine, {src1Desc, weight1Desc, biasDesc, Activation::None, PostOp::None});

    // Convolution 2: dst = activation(conv(src2, weight2) + dst)
    // We use dst as bias, which is supported by CUTLASS
    conv2 = newCUDAConv(engine, {src2Desc, weight2Desc, dstDesc, activation, PostOp::None});
  }

  bool CUDAConcatConv::isSupported() const
  {
    return conv1->isSupported() && conv2->isSupported();
  }

  size_t CUDAConcatConv::getScratchByteSize() const
  {
    assert(isSupported());
    return max(conv1->getScratchByteSize(), conv2->getScratchByteSize());
  }

  void CUDAConcatConv::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    conv1->setScratch(scratch);
    conv2->setScratch(scratch);
  }

  void CUDAConcatConv::updateSrc()
  {
    conv1->setSrc(src1);
    conv2->setSrc(src2);
  }

  void CUDAConcatConv::updateWeight()
  {
    if (finalized)
      throw std::logic_error("concatenation+convolution weight cannot be set after finalization");
  }

  void CUDAConcatConv::updateBias()
  {
    conv1->setBias(bias);
  }
  
  void CUDAConcatConv::updateDst()
  {
    conv1->setDst(dst);

    conv2->setBias(dst);
    conv2->setDst(dst);
  }

  void CUDAConcatConv::finalize()
  {
    assert(isSupported());

    // Split weight into weight1 and weight2
    auto weight1 = engine->newTensor(weight1Desc);
    auto weight2 = engine->newTensor(weight2Desc);

    auto weightHost  = weight->map(Access::Read);
    auto weight1Host = weight1->map(Access::WriteDiscard);
    auto weight2Host = weight2->map(Access::WriteDiscard);

    TensorAccessor4D<half, TensorLayout::ohwi> weightAcc  = *weightHost;
    TensorAccessor4D<half, TensorLayout::ohwi> weight1Acc = *weight1Host;
    TensorAccessor4D<half, TensorLayout::ohwi> weight2Acc = *weight2Host;

    for (int o = 0; o < weightAcc.O; ++o)
    {
      for (int h = 0; h < weightAcc.H; ++h)
      {
        for (int w = 0; w < weightAcc.W; ++w)
        {
          for (int i = 0; i < weight1Acc.I; ++i)
            weight1Acc(o, i, h, w) = weightAcc(o, i, h, w);

          for (int i = 0; i < weight2Acc.I; ++i)
            weight2Acc(o, i, h, w) = weightAcc(o, weight1Acc.I + i, h, w);
        }
      }
    }

    weight.reset();

    conv1->setWeight(weight1);
    conv1->finalize();

    conv2->setWeight(weight2);
    conv2->finalize();

    finalized = true;
  }

  void CUDAConcatConv::submit()
  {
    if (!finalized)
      throw std::logic_error("concatenation+convolution not finalized");

    conv1->submit();
    conv2->submit();
  }

} // namespace oidn