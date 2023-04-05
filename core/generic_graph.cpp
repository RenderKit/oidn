// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "generic_graph.h"
#include "concat_conv_chw.h"
#include "concat_conv_hwc.h"
#include "tensor_reorder.h"

OIDN_NAMESPACE_BEGIN

  GenericGraph::GenericGraph(const Ref<Engine>& engine,
                             const std::shared_ptr<TensorMap>& constTensors)
    : engine(engine),
      constTensors(constTensors) {}

  std::shared_ptr<InputProcess> GenericGraph::addInputProcess(const std::string& name,
                                                              const TensorDims& srcDims,
                                                              int tileAlignment,
                                                              const std::shared_ptr<TransferFunction>& transferFunc,
                                                              bool hdr,
                                                              bool snorm)
  {
    auto op = engine->newInputProcess({srcDims, tileAlignment, transferFunc, hdr, snorm});
    op->setName(name);
    TensorAlloc* dstAlloc = addOp(op, {}, op->getDstDesc());

    lazyInits.push_back([=]()
    {
      op->setDst(dstAlloc->tensor);
    });

    return op;
  }

  std::shared_ptr<OutputProcess> GenericGraph::addOutputProcess(const std::string& name,
                                                                const std::shared_ptr<Op>& srcOp,
                                                                const std::shared_ptr<TransferFunction>& transferFunc,
                                                                bool hdr,
                                                                bool snorm)
  {
    TensorAlloc* srcAlloc = tensorAllocsByOp[srcOp.get()];
    const TensorDesc srcDesc = srcAlloc->desc;
    auto op = engine->newOutputProcess({srcDesc, transferFunc, hdr, snorm});
    op->setName(name);
    addOp(op, {srcOp});

    lazyInits.push_back([=]()
    {
      op->setSrc(srcAlloc->tensor);
    });

    return op;
  }

  std::shared_ptr<Op> GenericGraph::addConv(const std::string& name,
                                            const std::shared_ptr<Op>& srcOp,
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
                                  engine->getDevice()->getTensorDataType()};

    TensorDesc finalBiasDesc = {bias->getDims(),
                                {round_up(bias->getX(), blockC)},
                                TensorLayout::x,
                                engine->getDevice()->getTensorDataType()};

    TensorAlloc* srcAlloc = tensorAllocsByOp[srcOp.get()];
    const TensorDesc srcDesc = srcAlloc->desc;
    auto conv = engine->newConv({srcDesc, finalWeightDesc, finalBiasDesc, activation, postOp});
    conv->setName(name);
    TensorAlloc* dstAlloc = addOp(conv, {srcOp}, conv->getDstDesc());

    lazyInits.push_back([=]()
    {
      conv->setSrc(srcAlloc->tensor);
      conv->setDst(dstAlloc->tensor);

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

    constByteSize += finalWeightDesc.getByteSize() + finalBiasDesc.getByteSize();
    return conv;
  }

  std::shared_ptr<ConcatConv> GenericGraph::addConcatConv(const std::string& name,
                                                          const std::shared_ptr<Op>& src1Op,
                                                          const std::shared_ptr<Op>& src2Op,
                                                          Activation activation)
  {
    auto weight = (*constTensors)[name + ".weight"];
    auto bias   = (*constTensors)[name + ".bias"];

    if (weight->getRank() != 4 || bias->getRank() != 1)
      throw std::invalid_argument("invalid convolution weight/bias");

    const int blockC = engine->getDevice()->getTensorBlockC();

    TensorAlloc* src1Alloc = tensorAllocsByOp[src1Op.get()];
    TensorAlloc* src2Alloc = tensorAllocsByOp[src2Op.get()];
    const TensorDesc src1Desc = src1Alloc->desc;
    const TensorDesc src2Desc = src2Alloc->desc;

    TensorDims finalWeightDims{round_up(weight->getO(), blockC),
                                src1Desc.getPaddedC() + src2Desc.getPaddedC(),
                                weight->getH(),
                                weight->getW()};

    TensorDesc finalWeightDesc = {weight->getDims(),
                                  finalWeightDims,
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
      TensorAlloc* dstAlloc = addOp(concatConv, {src1Op, src2Op}, concatConv->getDstDesc());

      lazyInits.push_back([=]()
      {
        concatConv->setSrc(src1Alloc->tensor, src2Alloc->tensor);
        concatConv->setDst(dstAlloc->tensor);

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

      constByteSize += concatConv->getWeight1Desc().getByteSize() +
                       concatConv->getWeight2Desc().getByteSize() + 
                       finalBiasDesc.getByteSize();
      return concatConv;
    }
    else
    {
      auto concatConv = std::make_shared<ConcatConvCHW>(engine, concatConvDesc);
      concatConv->setName(name);
      TensorAlloc* dstAlloc = addOp(concatConv, {src1Op, src2Op}, concatConv->getDstDesc(), true);

      lazyInits.push_back([=]()
      {
        concatConv->setSrc(src1Alloc->tensor, src2Alloc->tensor);
        concatConv->setDst(dstAlloc->tensor);

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

      constByteSize += finalWeightDesc.getByteSize() + finalBiasDesc.getByteSize();
      return concatConv;
    }
  }

  std::shared_ptr<Pool> GenericGraph::addPool(const std::string& name,
                                              const std::shared_ptr<Op>& srcOp)
  {
    TensorAlloc* srcAlloc = tensorAllocsByOp[srcOp.get()];
    const TensorDesc srcDesc = srcAlloc->desc;
    auto op = engine->newPool({srcDesc});
    op->setName(name);
    TensorAlloc* dstAlloc = addOp(op, {srcOp}, op->getDstDesc());

    lazyInits.push_back([=]()
    {
      op->setSrc(srcAlloc->tensor);
      op->setDst(dstAlloc->tensor);
    });

    return op;
  }

  std::shared_ptr<Upsample> GenericGraph::addUpsample(const std::string& name,
                                                      const std::shared_ptr<Op>& srcOp)
  {
    TensorAlloc* srcAlloc = tensorAllocsByOp[srcOp.get()];
    const TensorDesc srcDesc = srcAlloc->desc;
    auto op = engine->newUpsample({srcDesc});
    op->setName(name);
    TensorAlloc* dstAlloc = addOp(op, {srcOp}, op->getDstDesc());

    lazyInits.push_back([=]()
    {
      op->setSrc(srcAlloc->tensor);
      op->setDst(dstAlloc->tensor);
    });

    return op;
  }
  
  void GenericGraph::addOp(const std::shared_ptr<Op>& op,
                           const std::vector<std::shared_ptr<Op>>& srcOps,
                           bool concatSrcs)
  {
    if (finalized)
      throw std::logic_error("graph cannot be changed after finalization");

    const int opID = int(ops.size());

    TensorAlloc* prev = nullptr;
    for (const auto& srcOp : srcOps)
    {
      TensorAlloc* cur = tensorAllocsByOp[srcOp.get()];
      cur->lastOpID = opID;
      
      if (concatSrcs && prev)
      {
        if (cur->prev || prev->next)
          throw std::logic_error("invalid tensor allocation constraints");
        cur->prev = prev;
        prev->next = cur;
      }

      prev = cur;
    }
    
    ops.push_back(op);
    dirty = true;
  }

  GenericGraph::TensorAlloc* GenericGraph::addOp(const std::shared_ptr<Op>& op,
                                                 const std::vector<std::shared_ptr<Op>>& srcOps,
                                                 const TensorDesc& dstDesc, bool concatSrcs)
  {
    const int opID = int(ops.size());

    // Create a tensor allocation record for the destination of the operation
    tensorAllocs.emplace_back(new TensorAlloc(dstDesc, opID));
    TensorAlloc* dstAlloc = tensorAllocs.back().get();
    tensorAllocsByOp[op.get()] = dstAlloc;

    addOp(op, srcOps, concatSrcs);
    return dstAlloc;
  }

  void GenericGraph::planAllocations()
  {
    // Determine the chunks to allocate. Each chunk contains one or more tensors consecutively
    struct Chunk
    {
      TensorAlloc* firstAlloc;
      int firstOpID;
      int lastOpID;
      size_t byteSize;
    };

    std::vector<Chunk> chunks;
    
    // Iterate over all tensor allocations and find the first allocation in each chunk
    for (const auto& alloc : tensorAllocs)
    {
      // If the allocation is not the first in a chunk, skip it
      if (alloc->prev)
        continue;

      // Initialize the chunk
      Chunk chunk;
      chunk.firstAlloc = alloc.get();
      chunk.byteSize   = 0;
      chunk.firstOpID  = alloc->firstOpID;
      chunk.lastOpID   = alloc->lastOpID;

      // Iterate over all allocations in the chunk
      for (TensorAlloc* curAlloc = chunk.firstAlloc; curAlloc; curAlloc = curAlloc->next)
      {
        chunk.byteSize += curAlloc->byteSize;
        chunk.firstOpID = min(chunk.firstOpID, curAlloc->firstOpID);
        chunk.lastOpID  = max(chunk.lastOpID,  curAlloc->lastOpID);
      }

      chunks.push_back(chunk);
    }

    // Sort the chunks by size in descending order
    std::sort(chunks.begin(), chunks.end(),
              [](const Chunk& a, const Chunk& b) { return a.byteSize > b.byteSize; });

    // Track the active allocations sorted by offset in ascending order
    std::vector<TensorAlloc*> activeAllocs;
    tensorScratchByteSize = 0;

    // Iterate over the sorted chunks to allocate
    for (const Chunk& chunk : chunks)
    {
      size_t curByteOffset   = 0;
      size_t bestByteOffset  = SIZE_MAX;
      size_t bestGapByteSize = SIZE_MAX;

      // Iterate over the active allocations sorted by offset in ascending order
      // Find the smallest gap between them that is large enough to fit the chunk
      for (const TensorAlloc* alloc : activeAllocs)
      {
        // If the allocation does not overlap with the chunk in time, skip it
        if (alloc->lastOpID < chunk.firstOpID || alloc->firstOpID > chunk.lastOpID)
          continue;

        // Check whether the current gap is large enough to fit the chunk
        if (curByteOffset + chunk.byteSize <= alloc->byteOffset &&
            alloc->byteOffset - curByteOffset < bestGapByteSize)
        {
          bestByteOffset  = curByteOffset;
          bestGapByteSize = alloc->byteOffset - curByteOffset;
        }

        curByteOffset = max(curByteOffset, alloc->byteOffset + alloc->byteSize);
      }

      if (bestByteOffset == SIZE_MAX)
        bestByteOffset = curByteOffset;

      // Assign offsets to the allocations in the chunk, and add them to the sorted active allocations
      for (TensorAlloc* alloc = chunk.firstAlloc; alloc; alloc = alloc->next)
      {
        alloc->byteOffset = bestByteOffset;
        
        auto it = std::upper_bound(activeAllocs.begin(), activeAllocs.end(), alloc,
                    [](const TensorAlloc* a, const TensorAlloc* b) { return a->byteOffset < b->byteOffset; });
        activeAllocs.insert(it, alloc);

        bestByteOffset += alloc->byteSize;
      }

      tensorScratchByteSize = max(tensorScratchByteSize, bestByteOffset);
    }

    // Compute the size of the operation scratch
    size_t opScratchByteSize = 0;
    for (const auto& op : ops)
      opScratchByteSize = max(opScratchByteSize, op->getScratchAlignedSize());

    dirty = false;
  }

  double GenericGraph::getWorkAmount() const
  {
    return double(ops.size());
  }

  bool GenericGraph::isSupported() const
  {
    for (const auto& op : ops)
      if (!op->isSupported())
        return false;
    return true;
  }

  size_t GenericGraph::getScratchAlignedSize()
  {
    if (dirty)
      planAllocations();
    return opScratchByteSize + tensorScratchByteSize;
  }

  void GenericGraph::setScratch(const Ref<Buffer>& scratch)
  {
    this->scratch = scratch;
  }

  void GenericGraph::cleanup()
  {
    lazyInits.clear();
    tensorAllocsByOp.clear();
    tensorAllocs.clear();
  }

  void GenericGraph::clear()
  {
    cleanup();
    ops.clear();
    scratch.reset();
    opScratchByteSize = 0;
    tensorScratchByteSize = 0;
    constByteSize = 0;
    dirty = false;
  }

  void GenericGraph::finalize()
  {
    if (dirty)
      planAllocations();

    for (auto& tensorAlloc : tensorAllocs)
      tensorAlloc->tensor = scratch->newTensor(tensorAlloc->desc, opScratchByteSize + tensorAlloc->byteOffset);

    for (auto& lazyInit : lazyInits)
      lazyInit();
    lazyInits.clear();

    for (auto& op : ops)
    {
      op->setScratch(scratch);
      op->finalize();
    }

    cleanup();
    constTensors.reset();
    
    finalized = true;
  }

  void GenericGraph::run(Progress& progress)
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
