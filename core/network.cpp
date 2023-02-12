// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "network.h"
#include "conv.h"
#include "pool.h"
#include "upsample.h"
#include "color.h"

OIDN_NAMESPACE_BEGIN

  Network::Network(const Ref<Engine>& engine, const std::shared_ptr<Weights>& weights)
    : engine(engine),
      weights(weights) {}

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
    assert(weights);
    auto weight = weights->get(name + ".weight");
    auto bias   = weights->get(name + ".bias");

    auto op = engine->newConv({srcDesc, weight->getDesc(), bias->getDesc(), activation, postOp});
    op->setWeight(weight);
    op->setBias(bias);
    op->setName(name);
    ops.push_back(op);
    return op;
  }

   std::shared_ptr<ConcatConv> Network::addConcatConv(const std::string& name,
                                                      const TensorDesc& src1Desc,
                                                      const TensorDesc& src2Desc,
                                                      Activation activation)
  {
    assert(weights);
    auto weight = weights->get(name + ".weight");
    auto bias   = weights->get(name + ".bias");

    auto op = engine->newConcatConv({src1Desc, src2Desc, weight->getDesc(), bias->getDesc(), activation});
    op->setWeight(weight);
    op->setBias(bias);
    op->setName(name);
    ops.push_back(op);
    return op;
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
  }

  void Network::finalize()
  {
    for (auto& op : ops)
      op->finalize();

    weights.reset();
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

OIDN_NAMESPACE_END
