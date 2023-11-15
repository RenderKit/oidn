// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//#define OIDN_MICROBENCH 1000 // number of microbenchmark iterations

#include "graph.h"
#include "concat_conv_chw.h"
#include "concat_conv_hwc.h"
#include "tensor_reorder.h"
#if defined(OIDN_MICROBENCH)
  #include "common/timer.h"
#endif

OIDN_NAMESPACE_BEGIN

  Graph::Graph(const Ref<Engine>& engine,
               const std::shared_ptr<TensorMap>& constTensors,
               bool fastMath)
    : engine(engine),
      constTensors(constTensors),
      fastMath(fastMath) {}

  Ref<InputProcess> Graph::addInputProcess(const std::string& name,
                                           const TensorDims& srcDims,
                                           int tileAlignment,
                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                           bool hdr,
                                           bool snorm)
  {
    auto op = engine->newInputProcess({srcDims, tileAlignment, transferFunc, hdr, snorm});
    op->setName(name);
    auto dstAlloc = addOp(op, {}, op->getDstDesc());

    lazyInits.push_back([=]()
    {
      op->setDst(dstAlloc->tensor);
    });

    return op;
  }

  Ref<OutputProcess> Graph::addOutputProcess(const std::string& name,
                                             const Ref<Op>& srcOp,
                                             const std::shared_ptr<TransferFunction>& transferFunc,
                                             bool hdr,
                                             bool snorm)
  {
    auto srcAlloc = tensorAllocs[srcOp.get()];
    auto op = engine->newOutputProcess({srcAlloc->desc, transferFunc, hdr, snorm});
    op->setName(name);
    addOp(op, {srcOp});

    lazyInits.push_back([=]()
    {
      op->setSrc(srcAlloc->tensor);
    });

    return op;
  }

  Ref<Op> Graph::addConv(const std::string& name,
                         const Ref<Op>& srcOp,
                         Activation activation,
                         PostOp postOp)
  {
    if (postOp != PostOp::None && !engine->isConvSupported(postOp))
    {
      // If the engine does not support the specified fused convolution, split it into two ops
      auto conv = addConv(name, srcOp, activation, PostOp::None);
      switch (postOp)
      {
      case PostOp::Pool:
        return addPool(name + "_pool", conv);
      case PostOp::Upsample:
        return addUpsample(name + "_upsample", conv);
      default:
        throw std::invalid_argument("cannot split fused convolution");
      }
    }

    auto weight = (*constTensors)[name + ".weight"];
    auto bias   = (*constTensors)[name + ".bias"];

    if (weight->getRank() != 4 || bias->getRank() != 1)
      throw std::invalid_argument("invalid convolution weight/bias");

    const int blockC = engine->getDevice()->getTensorBlockC();

    TensorDims finalWeightDims{round_up(weight->getO(), blockC),
                               round_up(weight->getI(), blockC),
                               weight->getH(),
                               weight->getW()};

    TensorDesc finalWeightDesc = {weight->getDims(),
                                  finalWeightDims,
                                  engine->getDevice()->getWeightLayout(),
                                  engine->getDevice()->getWeightDataType()};

    TensorDesc finalBiasDesc = {bias->getDims(),
                                {round_up(bias->getX(), blockC)},
                                TensorLayout::x,
                                engine->getDevice()->getTensorDataType()};

    auto srcAlloc = tensorAllocs[srcOp.get()];
    auto conv = engine->newConv({srcAlloc->desc, finalWeightDesc, finalBiasDesc, activation, postOp, fastMath});
    conv->setName(name);
    auto dstAlloc = addOp(conv, {srcOp}, conv->getDstDesc());

    lazyInits.push_back([=]()
    {
      conv->setSrc(srcAlloc->tensor);
      conv->setDst(dstAlloc->tensor);

      // Reorder the weight tensor
      auto finalWeight = makeRef<HostTensor>(finalWeightDesc);
      reorderWeight(*weight, *finalWeight);
      conv->setWeight(finalWeight);

      // Reorder the bias tensor
      auto finalBias = makeRef<HostTensor>(finalBiasDesc);
      reorderBias(*bias, *finalBias);
      conv->setBias(finalBias);
    });

    privateByteSize += finalWeightDesc.getByteSize() + finalBiasDesc.getByteSize();
    return conv;
  }

  Ref<Op> Graph::addConcatConv(const std::string& name,
                               const Ref<Op>& src1Op,
                               const Ref<Op>& src2Op,
                               Activation activation)
  {
    auto weight = (*constTensors)[name + ".weight"];
    auto bias   = (*constTensors)[name + ".bias"];

    if (weight->getRank() != 4 || bias->getRank() != 1)
      throw std::invalid_argument("invalid convolution weight/bias");

    const int blockC = engine->getDevice()->getTensorBlockC();

    auto src1Alloc = tensorAllocs[src1Op.get()];
    auto src2Alloc = tensorAllocs[src2Op.get()];
    const TensorDesc src1Desc = src1Alloc->desc;
    const TensorDesc src2Desc = src2Alloc->desc;

    TensorDims finalWeightDims{round_up(weight->getO(), blockC),
                               src1Desc.getPaddedC() + src2Desc.getPaddedC(),
                               weight->getH(),
                               weight->getW()};

    TensorDesc finalWeightDesc = {weight->getDims(),
                                  finalWeightDims,
                                  engine->getDevice()->getWeightLayout(),
                                  engine->getDevice()->getWeightDataType()};

    TensorDesc finalBiasDesc = {bias->getDims(),
                                {round_up(bias->getX(), blockC)},
                                TensorLayout::x,
                                engine->getDevice()->getTensorDataType()};

    ConcatConvDesc concatConvDesc{src1Desc, src2Desc, finalWeightDesc, finalBiasDesc, activation, fastMath};

    if (engine->getDevice()->getTensorLayout() == TensorLayout::hwc)
    {
      auto concatConv = makeRef<ConcatConvHWC>(engine, concatConvDesc);
      concatConv->setName(name);
      auto dstAlloc = addOp(concatConv, {src1Op, src2Op}, concatConv->getDstDesc());

      lazyInits.push_back([=]()
      {
        concatConv->setSrc(src1Alloc->tensor, src2Alloc->tensor);
        concatConv->setDst(dstAlloc->tensor);

        // Reorder the weight tensor
        auto finalWeight1 = makeRef<HostTensor>(concatConv->getWeight1Desc());
        auto finalWeight2 = makeRef<HostTensor>(concatConv->getWeight2Desc());

        reorderWeight(*weight, 0, src1Desc.getC(),
                      *finalWeight1, 0, src1Desc.getPaddedC());
        reorderWeight(*weight, src1Desc.getC(), src2Desc.getC(),
                      *finalWeight2, 0, src2Desc.getPaddedC());

        concatConv->setWeight(finalWeight1, finalWeight2);

        // Reorder the bias tensor
        auto finalBias = makeRef<HostTensor>(finalBiasDesc);
        reorderBias(*bias, *finalBias);
        concatConv->setBias(finalBias);
      });

      privateByteSize += concatConv->getWeight1Desc().getByteSize() +
                         concatConv->getWeight2Desc().getByteSize() +
                         finalBiasDesc.getByteSize();
      return concatConv;
    }
    else
    {
      auto concatConv = makeRef<ConcatConvCHW>(engine, concatConvDesc);
      concatConv->setName(name);
      auto dstAlloc = addOp(concatConv, {src1Op, src2Op}, concatConv->getDstDesc(), true);

      lazyInits.push_back([=]()
      {
        concatConv->setSrc(src1Alloc->tensor, src2Alloc->tensor);
        concatConv->setDst(dstAlloc->tensor);

        // Reorder the weight tensor
        auto finalWeight = makeRef<HostTensor>(finalWeightDesc);

        reorderWeight(*weight, 0, src1Desc.getC(),
                      *finalWeight, 0, src1Desc.getPaddedC());
        reorderWeight(*weight, src1Desc.getC(), src2Desc.getC(),
                      *finalWeight, src1Desc.getPaddedC(), src2Desc.getPaddedC());

        concatConv->setWeight(finalWeight);

        // Reorder the bias tensor
        auto finalBias = makeRef<HostTensor>(finalBiasDesc);
        reorderBias(*bias, *finalBias);
        concatConv->setBias(finalBias);
      });

      privateByteSize += finalWeightDesc.getByteSize() + finalBiasDesc.getByteSize();
      return concatConv;
    }
  }

  Ref<Op> Graph::addPool(const std::string& name,
                         const Ref<Op>& srcOp)
  {
    auto srcAlloc = tensorAllocs[srcOp.get()];
    auto op = engine->newPool({srcAlloc->desc});
    op->setName(name);
    auto dstAlloc = addOp(op, {srcOp}, op->getDstDesc());

    lazyInits.push_back([=]()
    {
      op->setSrc(srcAlloc->tensor);
      op->setDst(dstAlloc->tensor);
    });

    return op;
  }

  Ref<Op> Graph::addUpsample(const std::string& name,
                             const Ref<Op>& srcOp)
  {
    auto srcAlloc = tensorAllocs[srcOp.get()];
    auto op = engine->newUpsample({srcAlloc->desc});
    op->setName(name);
    auto dstAlloc = addOp(op, {srcOp}, op->getDstDesc());

    lazyInits.push_back([=]()
    {
      op->setSrc(srcAlloc->tensor);
      op->setDst(dstAlloc->tensor);
    });

    return op;
  }

  void Graph::addOp(const Ref<Op>& op,
                    const std::vector<Ref<Op>>& srcOps,
                    bool concatSrcs)
  {
    if (finalized)
      throw std::logic_error("graph cannot be changed after finalization");

    const int opID = int(ops.size());

    // Add the source tensor allocations as dependencies for the operation
    std::vector<int> srcAllocIDs;
    for (const auto& srcOp : srcOps)
      srcAllocIDs.push_back(tensorAllocs[srcOp.get()]->id);
    tensorScratchPlanner.addDepAllocs(opID, srcAllocIDs, concatSrcs);

    ops.push_back(op);
    dirty = true;
  }

  std::shared_ptr<Graph::TensorAlloc> Graph::addOp(
                                        const Ref<Op>& op,
                                        const std::vector<Ref<Op>>& srcOps,
                                        const TensorDesc& dstDesc,
                                        bool concatSrcs)
  {
    const int opID = int(ops.size());

    // Create a tensor allocation record for the destination of the operation
    const auto dstByteSizeAndAlignment =
      engine->getBufferByteSizeAndAlignment(dstDesc.getByteSize(), Storage::Device);
    const int dstAllocID = tensorScratchPlanner.newAlloc(opID, dstByteSizeAndAlignment);
    auto dstAlloc = std::make_shared<TensorAlloc>(dstDesc, dstAllocID);
    tensorAllocs[op.get()] = dstAlloc;

    addOp(op, srcOps, concatSrcs);
    return dstAlloc;
  }

  void Graph::planAllocs()
  {
    tensorScratchPlanner.commit();

    // Compute the size of the operation scratch
    size_t opScratchByteSize = 0;
    for (const auto& op : ops)
      opScratchByteSize = max(opScratchByteSize, op->getScratchByteSize());
    opScratchByteSize = round_up(opScratchByteSize, tensorScratchPlanner.getByteAlignment());

    // Compute the size of the tensor scratch
    tensorScratchByteOffset = opScratchByteSize;
    size_t tensorScratchByteSize = round_up(tensorScratchPlanner.getByteSize(), memoryAlignment);

    // Compute the total scratch size
    scratchByteSize = opScratchByteSize + tensorScratchByteSize;

    dirty = false;
  }

  double Graph::getWorkAmount() const
  {
    return double(ops.size());
  }

  bool Graph::isSupported() const
  {
    for (const auto& opTensorAllocPair : tensorAllocs)
      if (!engine->isSupported(opTensorAllocPair.second->desc))
        return false;

    for (const auto& op : ops)
      if (!op->isSupported())
        return false;

    return true;
  }

  size_t Graph::getScratchByteSize()
  {
    if (dirty)
      planAllocs();
    return scratchByteSize;
  }

  void Graph::setScratch(const Ref<Buffer>& scratch)
  {
    if (scratch->getByteSize() < getScratchByteSize())
      throw std::invalid_argument("graph scratch buffer is too small");
    this->scratch = scratch;
  }

  void Graph::cleanup()
  {
    lazyInits.clear();
    tensorAllocs.clear();
    tensorScratchPlanner.clear();
  }

  void Graph::clear()
  {
    if (finalized)
      throw std::logic_error("graph cannot be cleared after finalization");

    cleanup();
    ops.clear();
    scratch.reset();
    scratchByteSize = 0;
    privateByteSize = 0;
    tensorScratchByteOffset = 0;
    dirty = false;
  }

  void Graph::finalize()
  {
    if (dirty)
      planAllocs();

    for (const auto& opTensorAllocPair : tensorAllocs)
    {
      auto& alloc = opTensorAllocPair.second;
      const size_t byteOffset = tensorScratchPlanner.getAllocByteOffset(alloc->id);
      alloc->tensor = scratch->newTensor(alloc->desc, tensorScratchByteOffset + byteOffset);
    }

    for (auto& lazyInit : lazyInits)
      lazyInit();

    for (auto& op : ops)
    {
      op->setScratch(scratch);
      op->finalize();
    }

    cleanup();
    constTensors.reset();

    finalized = true;
  }

  void Graph::run(Progress& progress)
  {
    if (!finalized)
      throw std::logic_error("graph not finalized");

  #if defined(OIDN_MICROBENCH)
    double totalTime = 0;
    std::cerr << std::endl;
    std::cerr << "op,name,msec" << std::endl;
  #endif

    for (size_t i = 0; i < ops.size(); ++i)
    {
      ops[i]->submit();

    #if defined(OIDN_MICROBENCH)
      engine->wait();
      const int numRuns = OIDN_MICROBENCH;
      Timer timer;
      for (int j = 0; j < numRuns; ++j)
        ops[i]->submit();
      engine->wait();
      const double time = timer.query() / numRuns;
      std::cerr << i << "," << ops[i]->getName() << "," << time * 1000 << std::endl;
      totalTime += time;
    #endif

    #if 0
      // Dump
      engine->wait();
      Ref<Tensor> dst;

      if (auto conv = std::dynamic_pointer_cast<Conv>(ops[i]))
        dst = conv->getDst();
      else if (auto conv = std::dynamic_pointer_cast<ConcatConv>(ops[i]))
        dst = conv->getDst();
      else if (auto pool = std::dynamic_pointer_cast<Pool>(ops[i]))
        dst = pool->getDst();
      else if (auto upsample = std::dynamic_pointer_cast<Upsample>(ops[i]))
        dst = upsample->getDst();

      if (dst)
      {
        std::cout << std::setfill('0') << std::setw(2) << i << ": "
                  << std::hex << std::setfill('0') << std::setw(8) << dst->getHash() << std::dec
                  << " " << ops[i]->getName() << std::endl;

        dst->dump(toString(i) + "_" + ops[i]->getName() + "_");
      }
    #endif

      progress.update(engine, 1);
    }

  #if defined(OIDN_MICROBENCH)
    std::cerr << ",total," << totalTime * 1000 << std::endl;
  #endif
  }

OIDN_NAMESPACE_END
