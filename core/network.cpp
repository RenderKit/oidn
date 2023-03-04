// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "network.h"
#include "conv.h"
#include "concat_conv_chw.h"
#include "concat_conv_hwc.h"
#include "pool.h"
#include "upsample.h"
#include "color.h"
#include "tza.h"

OIDN_NAMESPACE_BEGIN

  Network::Network(const Ref<Engine>& engine, const Data& weightsBlob)
    : engine(engine),
      weights(parseTZA(engine, weightsBlob.ptr, weightsBlob.size)) {}

  std::shared_ptr<InputProcess> Network::addInputProcess(const std::string& name,
                                                         const TensorDims& srcDims,
                                                         int alignment,
                                                         const std::shared_ptr<TransferFunction>& transferFunc,
                                                         bool hdr,
                                                         bool snorm)
  {
    auto op = engine->newInputProcess({srcDims, alignment, transferFunc, hdr, snorm});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  std::shared_ptr<OutputProcess> Network::addOutputProcess(const std::string& name,
                                                           const TensorDesc& srcDesc,
                                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                                           bool hdr,
                                                           bool snorm)
  {
    auto op = engine->newOutputProcess({srcDesc, transferFunc, hdr, snorm});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  std::shared_ptr<Conv> Network::addConv(const std::string& name,
                                         const TensorDesc& srcDesc,
                                         Activation activation,
                                         PostOp postOp)
  {
    auto weight = weights[name + ".weight"];
    auto bias   = weights[name + ".bias"];

    if (weight->getRank() != 4 || bias->getRank() != 1)
      throw std::invalid_argument("invalid convolution weight/bias");

    const int blockC = engine->getDevice()->getTensorBlockC();

    TensorDims paddedWeightDims{round_up(weight->getO(), blockC),
                                round_up(weight->getI(), blockC),
                                weight->getH(),
                                weight->getW()};

    TensorDesc finalWeightDesc = {weight->getDims(),
                                  paddedWeightDims,
                                  engine->getDevice()->getWeightLayout(),
                                  engine->getDevice()->getTensorDataType()};

    TensorDesc finalBiasDesc = {bias->getDims(),
                                {round_up(bias->getX(), blockC)},
                                TensorLayout::x,
                                engine->getDevice()->getTensorDataType()};

    auto conv = engine->newConv({srcDesc, finalWeightDesc, finalBiasDesc, activation, postOp});
    conv->setName(name);
    ops.push_back(conv);

    lazyInits.push_back([=]()
    {
      // Reorder the weight tensor
      auto finalWeight = engine->newTensor(finalWeightDesc);
      reorderWeight(*weight, 0, weight->getI(),
                    *finalWeight->map(Access::WriteDiscard), 0, finalWeight->getPaddedI());
      conv->setWeight(finalWeight);

      // Reorder the bias tensor
      auto finalBias = engine->newTensor(finalBiasDesc);
      reorderBias(*bias, *finalBias->map(Access::WriteDiscard));
      conv->setBias(finalBias);
    });

    privateByteSize += finalWeightDesc.getByteSize() + finalBiasDesc.getByteSize();
    return conv;
  }

  std::shared_ptr<ConcatConv> Network::addConcatConv(const std::string& name,
                                                     const TensorDesc& src1Desc,
                                                     const TensorDesc& src2Desc,
                                                     Activation activation)
  {
    auto weight = weights[name + ".weight"];
    auto bias   = weights[name + ".bias"];

    if (weight->getRank() != 4 || bias->getRank() != 1)
      throw std::invalid_argument("invalid convolution weight/bias");

    const int blockC = engine->getDevice()->getTensorBlockC();

    TensorDims paddedWeightDims{round_up(weight->getO(), blockC),
                                src1Desc.getPaddedC() + src2Desc.getPaddedC(),
                                weight->getH(),
                                weight->getW()};

    TensorDesc finalWeightDesc = {weight->getDims(),
                                  paddedWeightDims,
                                  engine->getDevice()->getWeightLayout(),
                                  engine->getDevice()->getTensorDataType()};

    TensorDesc finalBiasDesc = {bias->getDims(),
                                {round_up(bias->getX(), blockC)},
                                TensorLayout::x,
                                engine->getDevice()->getTensorDataType()};

    ConcatConvDesc concatConvDesc{src1Desc, src2Desc, finalWeightDesc, finalBiasDesc, activation};

    if (engine->getDevice()->getTensorLayout() == TensorLayout::hwc)
    {
      auto concatConv = std::make_shared<ConcatConvHWC>(engine, concatConvDesc);
      concatConv->setName(name);
      ops.push_back(concatConv);

      lazyInits.push_back([=]()
      {
        // Reorder the weight tensor
        auto finalWeight1 = engine->newTensor(concatConv->getWeight1Desc());
        auto finalWeight2 = engine->newTensor(concatConv->getWeight2Desc());

        reorderWeight(*weight, 0, src1Desc.getC(),
                      *finalWeight1->map(Access::WriteDiscard), 0, src1Desc.getPaddedC());
        reorderWeight(*weight, src1Desc.getC(), src2Desc.getC(),
                      *finalWeight2->map(Access::WriteDiscard), 0, src2Desc.getPaddedC());

        concatConv->setWeight(finalWeight1, finalWeight2);

        // Reorder the bias tensor
        auto finalBias = engine->newTensor(finalBiasDesc);
        reorderBias(*bias, *finalBias->map(Access::WriteDiscard));
        concatConv->setBias(finalBias);
      });

      privateByteSize += concatConv->getWeight1Desc().getByteSize() +
                         concatConv->getWeight2Desc().getByteSize() + 
                         finalBiasDesc.getByteSize();
      return concatConv;
    }
    else
    {
      auto concatConv = std::make_shared<ConcatConvCHW>(engine, concatConvDesc);
      concatConv->setName(name);
      ops.push_back(concatConv);

      lazyInits.push_back([=]()
      {
        // Reorder the weight tensor
        auto finalWeight = engine->newTensor(finalWeightDesc);

        {
          auto finalWeightHost = finalWeight->map(Access::WriteDiscard);
          reorderWeight(*weight, 0, src1Desc.getC(),
                        *finalWeightHost, 0, src1Desc.getPaddedC());
          reorderWeight(*weight, src1Desc.getC(), src2Desc.getC(),
                        *finalWeightHost, src1Desc.getPaddedC(), src2Desc.getPaddedC());
        }

        concatConv->setWeight(finalWeight);

        // Reorder the bias tensor
        auto finalBias = engine->newTensor(finalBiasDesc);
        reorderBias(*bias, *finalBias->map(Access::WriteDiscard));
        concatConv->setBias(finalBias);
      });

      privateByteSize += finalWeightDesc.getByteSize() + finalBiasDesc.getByteSize();
      return concatConv;
    }
  }

  std::shared_ptr<Pool> Network::addPool(const std::string& name,
                                         const TensorDesc& srcDesc)
  {
    auto op = engine->newPool({srcDesc});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  std::shared_ptr<Upsample> Network::addUpsample(const std::string& name,
                                                 const TensorDesc& srcDesc)
  {
    auto op = engine->newUpsample({srcDesc});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  double Network::getWorkAmount() const
  {
    return double(ops.size());
  }

  bool Network::isSupported() const
  {
    for (const auto& op : ops)
      if (!op->isSupported())
        return false;
    return true;
  }

  size_t Network::getScratchAlignedSize() const
  {
    size_t scratchAlignedSize = 0;
    for (const auto& op : ops)
      scratchAlignedSize = max(scratchAlignedSize, op->getScratchAlignedSize());
    return scratchAlignedSize;
  }

  void Network::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    for (auto& op : ops)
      op->setScratch(scratch);
  }

  void Network::clear()
  {
    ops.clear();
    lazyInits.clear();
    privateByteSize = 0;
  }

  void Network::finalize()
  {
    for (auto& lazyInit : lazyInits)
      lazyInit();
    lazyInits.clear();

    for (auto& op : ops)
      op->finalize();

    weights.clear();
  }

  void Network::run(Progress& progress)
  {
    for (size_t i = 0; i < ops.size(); ++i)
    {
      ops[i]->submit();
      
    #if 0
      // Dump
      engine->wait();
      std::shared_ptr<Tensor> dst;

      if (auto conv = std::dynamic_pointer_cast<Conv>(ops[i]))
        dst = conv->getDst();
      else if (auto conv = std::dynamic_pointer_cast<ConcatConv>(ops[i]))
        dst = conv->getDst();
      else if (auto pool = std::dynamic_pointer_cast<Pool>(ops[i]))
        dst = pool->getDst();
      else if (auto upsample = std::dynamic_pointer_cast<Upsample>(ops[i]))
        dst = upsample->getDst();

      if (dst)
        dst->dump(toString(i) + "_" + ops[i]->getName() + "_");
    #endif

      progress.update(engine, 1);
    }
  }

  template<typename SrcT, typename DstT, TensorLayout srcLayout, TensorLayout dstLayout>
  bool Network::tryReorderWeight(const Tensor& src, int srcBeginI, int srcI, Tensor& dst, int dstBeginI, int dstI)
  {   
    assert(srcBeginI + srcI <= src.getPaddedI());
    assert(dstBeginI + dstI <= dst.getPaddedI());

    if (src.getDataType() != DataTypeOf<SrcT>::value || src.getLayout() != srcLayout ||
        dst.getDataType() != DataTypeOf<DstT>::value || dst.getLayout() != dstLayout)
      return false;
  
    TensorAccessor4D<SrcT, srcLayout> srcAcc = src;
    TensorAccessor4D<DstT, dstLayout> dstAcc = dst;

    for (int o = 0; o < dstAcc.O; ++o)
    {
      for (int i = 0; i < dstI; ++i)
      {
        for (int h = 0; h < dstAcc.H; ++h)
        {
          for (int w = 0; w < dstAcc.W; ++w)
          {
            SrcT value;
            if (o < srcAcc.O && i < srcI)
              value = srcAcc(o, srcBeginI + i, h, w);
            else
              value = 0; // padding

            dstAcc(o, dstBeginI + i, h, w) = DstT(value);
          }
        }
      }
    }

    return true;
  }

  void Network::reorderWeight(const Tensor& src, int srcBeginI, int srcI, Tensor& dst, int dstBeginI, int dstI)
  {
    bool ok =
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::oihw>        (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::oihw>        (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::OIhw8i8o>    (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::OIhw8i8o>    (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::OIhw16i16o>  (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::OIhw16i16o>  (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::OIhw2o8i8o2i>(src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::OIhw8i16o2i> (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, half,  TensorLayout::oihw, TensorLayout::ohwi>        (src, srcBeginI, srcI, dst, dstBeginI, dstI) ||
      tryReorderWeight<half, float, TensorLayout::oihw, TensorLayout::ohwi>        (src, srcBeginI, srcI, dst, dstBeginI, dstI);

    if (!ok)
      throw std::logic_error("unsupported weight layout or data type");
  }

  template<typename SrcT, typename DstT>
  bool Network::tryReorderBias(const Tensor& src, Tensor& dst)
  {
    if (src.getDataType() != DataTypeOf<SrcT>::value ||
        dst.getDataType() != DataTypeOf<DstT>::value)
      return false; 

    TensorAccessor1D<SrcT> srcAcc = src;
    TensorAccessor1D<DstT> dstAcc = dst;

    const int srcX = src.getX();

    for (int x = 0; x < srcX; ++x)
      dstAcc(x) = srcAcc(x);

    for (int x = srcX; x < dstAcc.X; ++x)
      dstAcc(x) = 0; // padding

    return true;
  }

  void Network::reorderBias(const Tensor& src, Tensor& dst)
  {
    bool ok = src.getLayout() == TensorLayout::x && dst.getLayout() == TensorLayout::x &&
      (tryReorderBias<half, half> (src, dst) ||
       tryReorderBias<half, float>(src, dst));

    if (!ok)
      throw std::logic_error("unsupported bias layout or data type");
  }

OIDN_NAMESPACE_END
