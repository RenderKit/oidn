// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

//#define OIDN_MICROBENCH 1000 // number of microbenchmark iterations

#include "graph.h"
#include "tensor_reorder.h"
#if defined(OIDN_MICROBENCH)
  #include "common/timer.h"
#endif

OIDN_NAMESPACE_BEGIN

  // Looks up a constant tensor by name. Using operator[] would silently insert a null tensor for
  // a name missing from a user-provided weights blob, which would then be dereferenced.
  static Ref<Tensor> getConstTensor(const TensorMap& constTensors, const std::string& name)
  {
    auto it = constTensors.find(name);
    if (it == constTensors.end() || !it->second)
      throw Exception(Error::InvalidOperation, "invalid or corrupted weights blob");
    return it->second;
  }

  Graph::Graph(Engine* engine,
               const std::shared_ptr<TensorMap>& constTensors,
               const std::shared_ptr<TensorMap>& cachedConstTensors,
               bool fastMath)
    : engine(engine),
      constTensors(constTensors),
      cachedConstTensors(cachedConstTensors),
      fastMath(fastMath) {}

  Ref<InputProcess> Graph::addInputProcess(const std::string& name,
                                           const TensorDims& srcDims,
                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                           bool hdr,
                                           bool snorm)
  {
    auto op = engine->newInputProcess({srcDims, transferFunc, hdr, snorm});
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
                         Fusion fusion)
  {
    if (fusion != Fusion::None && !engine->isConvSupported(fusion))
    {
      // If the engine does not support the specified fused convolution, split it into two ops
      switch (fusion)
      {
      case Fusion::UpsampleSrc0:
        {
           auto srcUpsample = addUpsample(name + "_upsample", srcOp);
           return addConv(name, srcUpsample, activation);
        }

      case Fusion::PoolDst:
        {
           auto conv = addConv(name, srcOp, activation);
           return addPool(name + "_pool", conv);
        }

      default:
        throw std::invalid_argument("unsupported convolution fusion");
      }
    }

    const std::string weightName = name + ".weight";
    const std::string biasName   = name + ".bias";
    Ref<Tensor> weight = getConstTensor(*constTensors, weightName);
    Ref<Tensor> bias   = getConstTensor(*constTensors, biasName);

    if (weight->getRank() != 4 || bias->getRank() != 1)
      throw std::invalid_argument("invalid convolution weight/bias");

    Device* device = engine->getDevice();
    const int blockC = device->getTensorBlockC();

    TensorDims finalWeightDims{round_up(weight->getO(), blockC),
                               round_up(weight->getI(), blockC),
                               weight->getH(),
                               weight->getW()};

    TensorDesc finalWeightDesc = {weight->getDims(),
                                  finalWeightDims,
                                  device->getWeightLayout(),
                                  device->getWeightDataType()};

    TensorDesc finalBiasDesc = {bias->getDims(),
                                {round_up(bias->getX(), blockC)},
                                TensorLayout::x,
                                device->getTensorDataType()};

    auto srcAlloc = tensorAllocs[srcOp.get()];
    auto conv = engine->newConv({srcAlloc->desc, finalWeightDesc, finalBiasDesc, activation, fusion, fastMath});
    conv->setName(name);
    auto dstAlloc = addOp(conv, {srcOp}, conv->getDstDesc());

    lazyInits.push_back([=]()
    {
      conv->setSrc(srcAlloc->tensor);
      conv->setDst(dstAlloc->tensor);

      Ref<Tensor> finalWeight = getCachedConstTensor(weightName, finalWeightDesc);
      if (!finalWeight)
      {
        finalWeight = makeRef<HostTensor>(finalWeightDesc);
        reorderWeight(*weight, *finalWeight);
        if (device->needWeightAndBiasOnDevice())
          finalWeight = finalWeight->toDevice(engine);
        setCachedConstTensor(weightName, finalWeight);
      }

      Ref<Tensor> finalBias = getCachedConstTensor(biasName, finalBiasDesc);
      if (!finalBias)
      {
        finalBias = makeRef<HostTensor>(finalBiasDesc);
        reorderBias(*bias, *finalBias);
        if (device->needWeightAndBiasOnDevice())
          finalBias = finalBias->toDevice(engine);
        setCachedConstTensor(biasName, finalBias);
      }

      conv->setWeight(finalWeight);
      conv->setBias(finalBias);
    });

    privateByteSize += finalWeightDesc.getByteSize() + finalBiasDesc.getByteSize();
    return conv;
  }

  Ref<Op> Graph::addConvPool(const std::string& name,
                             const Ref<Op>& srcOp,
                             Activation activation)
  {
    return addConv(name, srcOp, activation, Fusion::PoolDst);
  }

  Ref<Op> Graph::addConcatConv(const std::string& name,
                               const Ref<Op>& src0Op,
                               const Ref<Op>& src1Op,
                               Activation activation,
                               Fusion fusion)
  {
    Device* device = engine->getDevice();

    if (!engine->isConcatConvSupported(fusion) && !engine->isConcatConv2Supported(fusion))
    {
      // Need to split into separate upsample and concat+conv
      auto src0Upsample = addUpsample(name + "_upsample0", src0Op);
      return addConcatConv(name, src0Upsample, src1Op, activation);
    }

    const std::string weightName = name + ".weight";
    const std::string biasName   = name + ".bias";
    Ref<Tensor> weight = getConstTensor(*constTensors, weightName);
    Ref<Tensor> bias   = getConstTensor(*constTensors, biasName);

    if (weight->getRank() != 4 || bias->getRank() != 1)
      throw std::invalid_argument("invalid convolution weight/bias");

    const int blockC = device->getTensorBlockC();

    auto src0Alloc = tensorAllocs[src0Op.get()];
    auto src1Alloc = tensorAllocs[src1Op.get()];
    const TensorDesc src0Desc = src0Alloc->desc;
    const TensorDesc src1Desc = src1Alloc->desc;

    TensorDims finalWeightDims{round_up(weight->getO(), blockC),
                               src0Desc.getPaddedC() + src1Desc.getPaddedC(),
                               weight->getH(),
                               weight->getW()};

    TensorDesc finalWeightDesc = {weight->getDims(),
                                  finalWeightDims,
                                  device->getWeightLayout(),
                                  device->getWeightDataType()};

    TensorDesc finalBiasDesc = {bias->getDims(),
                                {round_up(bias->getX(), blockC)},
                                TensorLayout::x,
                                device->getTensorDataType()};

    ConcatConvDesc concatConvDesc{src0Desc, src1Desc, finalWeightDesc, finalBiasDesc, activation, fusion, fastMath};

    if (engine->isConcatConvSupported(fusion))
    {
      auto concatConv = engine->newConcatConv(concatConvDesc);
      concatConv->setName(name);
      auto dstAlloc = addOp(concatConv, {src0Op, src1Op}, concatConv->getDstDesc(),
                            concatConv->isPreConcatConv());

      lazyInits.push_back([=]()
      {
        concatConv->setSrc(src0Alloc->tensor, src1Alloc->tensor);
        concatConv->setDst(dstAlloc->tensor);

        Ref<Tensor> finalWeight = getCachedConstTensor(weightName, finalWeightDesc);
        if (!finalWeight)
        {
          finalWeight = makeRef<HostTensor>(finalWeightDesc);

          reorderWeight(*weight, 0, src0Desc.getC(),
                        *finalWeight, 0, src0Desc.getPaddedC());
          reorderWeight(*weight, src0Desc.getC(), src1Desc.getC(),
                        *finalWeight, src0Desc.getPaddedC(), src1Desc.getPaddedC());

          if (device->needWeightAndBiasOnDevice())
            finalWeight = finalWeight->toDevice(engine);

          setCachedConstTensor(weightName, finalWeight);
        }

        Ref<Tensor> finalBias = getCachedConstTensor(biasName, finalBiasDesc);
        if (!finalBias)
        {
          finalBias = makeRef<HostTensor>(finalBiasDesc);
          reorderBias(*bias, *finalBias);
          if (device->needWeightAndBiasOnDevice())
            finalBias = finalBias->toDevice(engine);
          setCachedConstTensor(biasName, finalBias);
        }

        concatConv->setWeight(finalWeight);
        concatConv->setBias(finalBias);
      });

      privateByteSize += finalWeightDesc.getByteSize() + finalBiasDesc.getByteSize();
      return concatConv;
    }
    else if (engine->isConcatConv2Supported(fusion))
    {
      // Concatenation + convolution is implemented as two separate convolutions
      auto concatConv = engine->newConcatConv2(concatConvDesc);
      concatConv->setName(name);
      auto dstAlloc = addOp(concatConv, {src0Op, src1Op}, concatConv->getDstDesc());

      lazyInits.push_back([=]()
      {
        concatConv->setSrc(src0Alloc->tensor, src1Alloc->tensor);
        concatConv->setDst(dstAlloc->tensor);

        const std::string weight0Name = weightName + "0";
        const std::string weight1Name = weightName + "1";
        Ref<Tensor> finalWeight0 = getCachedConstTensor(weight0Name, concatConv->getWeight0Desc());
        Ref<Tensor> finalWeight1 = getCachedConstTensor(weight1Name, concatConv->getWeight1Desc());

        if (!finalWeight0 || !finalWeight1)
        {
          finalWeight0 = makeRef<HostTensor>(concatConv->getWeight0Desc());
          finalWeight1 = makeRef<HostTensor>(concatConv->getWeight1Desc());

          reorderWeight(*weight, 0, src0Desc.getC(),
                        *finalWeight0, 0, src0Desc.getPaddedC());
          reorderWeight(*weight, src0Desc.getC(), src1Desc.getC(),
                        *finalWeight1, 0, src1Desc.getPaddedC());

          if (device->needWeightAndBiasOnDevice())
          {
            finalWeight0 = finalWeight0->toDevice(engine);
            finalWeight1 = finalWeight1->toDevice(engine);
          }

          setCachedConstTensor(weight0Name, finalWeight0);
          setCachedConstTensor(weight1Name, finalWeight1);
        }

        Ref<Tensor> finalBias = getCachedConstTensor(biasName, finalBiasDesc);
        if (!finalBias)
        {
          finalBias = makeRef<HostTensor>(finalBiasDesc);
          reorderBias(*bias, *finalBias);
          if (device->needWeightAndBiasOnDevice())
            finalBias = finalBias->toDevice(engine);
          setCachedConstTensor(biasName, finalBias);
        }

        concatConv->setWeight(finalWeight0, finalWeight1);
        concatConv->setBias(finalBias);
      });

      privateByteSize += concatConv->getWeight0Desc().getByteSize() +
                         concatConv->getWeight1Desc().getByteSize() +
                         finalBiasDesc.getByteSize();
      return concatConv;
    }
    else
    {
      throw std::logic_error("concat+conv operation is not implemented");
    }
  }

  Ref<Op> Graph::addUpsampleConcatConv(const std::string& name,
                                       const Ref<Op>& src0Op,
                                       const Ref<Op>& src1Op,
                                       Activation activation)
  {
    return addConcatConv(name, src0Op, src1Op, activation, Fusion::UpsampleSrc0);
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
    workAmount += op->getWorkAmount();
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
    workAmount = 0;
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
    cachedConstTensors.reset();

    finalized = true;
  }

  void Graph::submit(const Ref<Progress>& progress)
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
      ops[i]->submit(progress);

    #if defined(OIDN_MICROBENCH)
      engine->wait();
      const int numRuns = OIDN_MICROBENCH;
      Timer timer;
      for (int j = 0; j < numRuns; ++j)
        ops[i]->submit(progress);
      engine->wait();
      const double time = timer.query() / numRuns;
      std::cerr << i << "," << ops[i]->getName() << "," << time * 1000 << std::endl;
      totalTime += time;
    #endif

    #if 0
      // Dump
      engine->wait();

      Ref<Tensor> dst;

      if (auto inputProcess = dynamicRefCast<InputProcess>(ops[i]))
        dst = inputProcess->getDst();
      else if (auto conv = dynamicRefCast<Conv>(ops[i]))
        dst = conv->getDst();
      else if (auto conv = dynamicRefCast<ConcatConv>(ops[i]))
        dst = conv->getDst();
      else if (auto pool = dynamicRefCast<Pool>(ops[i]))
        dst = pool->getDst();
      else if (auto upsample = dynamicRefCast<Upsample>(ops[i]))
        dst = upsample->getDst();

      if (dst)
      {
        const uint32_t refHash = dst->getHash();

        std::cout << std::setfill('0') << std::setw(2) << i << ": "
                  << std::hex << std::setfill('0') << std::setw(8) << refHash << std::dec
                  << " " << ops[i]->getName() << std::endl;

        /*
        for (int j = 0; j < 100; ++j)
        {
          ops[i]->submit();
          engine->wait();
          const uint32_t hash = dst->getHash();
          if (hash != refHash)
            throw std::runtime_error("output hash mismatch (non-deterministic output)");
        }
        */

        dst->dump(toString(i) + "_" + ops[i]->getName() + "_");
      }
    #endif
    }

  #if defined(OIDN_MICROBENCH)
    std::cerr << ",total," << totalTime * 1000 << std::endl;
  #endif
  }

  Ref<Tensor> Graph::getCachedConstTensor(const std::string& name, const TensorDesc& desc)
  {
    if (cachedConstTensors)
    {
      auto tensorIter = cachedConstTensors->find(name);
      if (tensorIter != cachedConstTensors->end() && tensorIter->second->getDesc() == desc)
        return tensorIter->second;
    }

    return nullptr;
  }

  void Graph::setCachedConstTensor(const std::string& name, const Ref<Tensor>& tensor)
  {
    if (cachedConstTensors)
      (*cachedConstTensors)[name] = tensor;
  }

OIDN_NAMESPACE_END
