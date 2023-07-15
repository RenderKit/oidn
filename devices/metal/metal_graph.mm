// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_graph.h"
#include "metal_common.h"
#include "core/tensor_reorder.h"

OIDN_NAMESPACE_BEGIN

  MetalGraph::MetalGraph(const Ref<MetalEngine>& engine,
                         const std::shared_ptr<TensorMap>& constTensors)
    : engine(engine),
      constTensors(constTensors)
  {}

  MetalGraph::~MetalGraph()
  {
    if (graphExecDesc)
      [graphExecDesc release];
    if (graph)
      [graph release];
  }

  MetalGraph::TensorNode* MetalGraph::addOp(const std::shared_ptr<Op>& op, const TensorDesc& dstDesc)
  {
    if (finalized)
      throw std::logic_error("graph cannot be changed after finalization");
    if (op->getScratchByteSize() > 0)
      throw std::logic_error("scratch memory is not supported for MetalGraph ops");

    tensorNodes.emplace_back(new TensorNode{dstDesc, nullptr});
    TensorNode* dstNode = tensorNodes.back().get();
    tensorNodesByOp[op.get()] = dstNode;
    ops.push_back(op);
    return dstNode;
  }

  std::shared_ptr<InputProcess> MetalGraph::addInputProcess(
                                  const std::string& name,
                                  const TensorDims& srcDims,
                                  int tileAlignment,
                                  const std::shared_ptr<TransferFunction>& transferFunc,
                                  bool hdr,
                                  bool snorm)
  {
    if (!ops.empty())
      throw std::logic_error("input processing must be added first to the graph");

    inputProcess = engine->newInputProcess({srcDims, tileAlignment, transferFunc, hdr, snorm});
    inputProcess->setName(name);
    TensorNode* dstNode = addOp(inputProcess, inputProcess->getDstDesc());

    lazyInits.push_back([=]()
    {
      MPSGraphTensor* dst = toMPSGraphPlaceholder(graph, dstNode->desc);
      dstNode->tensor = dst;
      graphInput = dst;
      inputProcess->setDst(scratch->newTensor(dstNode->desc, 0));
    });

    scratchByteSize = max(scratchByteSize, dstNode->desc.getByteSize());
    return inputProcess;
  }

  std::shared_ptr<OutputProcess> MetalGraph::addOutputProcess(
                                   const std::string& name,
                                   const std::shared_ptr<Op>& srcOp,
                                   const std::shared_ptr<TransferFunction>& transferFunc,
                                   bool hdr,
                                   bool snorm)
  {
    if (!inputProcess || outputProcess)
      throw std::logic_error("output processing must be added last to the graph");

    TensorNode* srcNode = tensorNodesByOp[srcOp.get()];
    const TensorDesc srcDesc = srcNode->desc;
    outputProcess = engine->newOutputProcess({srcDesc, transferFunc, hdr, snorm});
    outputProcess->setName(name);
    addOp(outputProcess, srcDesc);

    lazyInits.push_back([=]()
    {
      graphOutput = srcNode->tensor;
      outputProcess->setSrc(scratch->newTensor(srcDesc, 0)); // alias the input tensor
    });

    scratchByteSize = max(scratchByteSize, srcDesc.getByteSize());
    return outputProcess;
  }

  std::shared_ptr<Op> MetalGraph::addConv(const std::string& name,
                                          const std::shared_ptr<Op>& srcOp,
                                          Activation activation,
                                          PostOp postOp)
  {
    if (!inputProcess || outputProcess)
      throw std::logic_error("op must be added to the graph between input and output processing");

    auto weight = (*constTensors)[name + ".weight"];
    auto bias   = (*constTensors)[name + ".bias"];

    if (weight->getRank() != 4 || bias->getRank() != 1)
      throw std::invalid_argument("invalid convolution weight/bias");

    TensorDesc finalWeightDesc = {weight->getDims(),
                                  engine->getDevice()->getWeightLayout(),
                                  engine->getDevice()->getTensorDataType()};

    TensorDesc finalBiasDesc = {bias->getDims(),
                                TensorLayout::x,
                                engine->getDevice()->getTensorDataType()};

    TensorNode* srcNode = tensorNodesByOp[srcOp.get()];
    const TensorDesc srcDesc = srcNode->desc;
    auto conv = engine->newConv({srcDesc, finalWeightDesc, finalBiasDesc, activation, postOp, false});
    conv->setName(name);
    const TensorDesc dstDesc = conv->getDstDesc();
    TensorNode* dstNode = addOp(conv, dstDesc);

    lazyInits.push_back([=]()
    {
      // Reorder the weight tensor
      auto finalHostWeight = std::make_shared<GenericTensor>(finalWeightDesc);
      reorderWeight(*weight, *finalHostWeight);

      // Reorder the bias tensor
      auto finalHostBias = std::make_shared<GenericTensor>(finalBiasDesc);
      reorderBias(*bias, *finalHostBias);

      auto finalWeight = toMPSGraphTensor(graph, finalHostWeight);
      auto finalBias   = toMPSGraphTensor(graph, finalHostBias);

      MPSGraphConvolution2DOpDescriptor* descr = [MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX: 1
                      strideInY: 1
                dilationRateInX: 1
                dilationRateInY: 1
                         groups: 1
                   paddingStyle: MPSGraphPaddingStyle::MPSGraphPaddingStyleTF_SAME
                     dataLayout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNHWC
                  weightsLayout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutOIHW
      ];

      auto dst = [graph convolution2DWithSourceTensor: srcNode->tensor
                                        weightsTensor: finalWeight
                                           descriptor: descr
                                                 name: nil];

      dst = [graph additionWithPrimaryTensor: dst
                             secondaryTensor: finalBias
                                        name: nil];

      if (activation == Activation::ReLU)
      {
        dst = [graph reLUWithTensor: dst
                               name: nil];
      }

      if (postOp == PostOp::Pool)
      {
        MPSGraphPooling2DOpDescriptor* descr = [MPSGraphPooling2DOpDescriptor
          descriptorWithKernelWidth: 2
                       kernelHeight: 2
                          strideInX: 2
                          strideInY: 2
                       paddingStyle: MPSGraphPaddingStyle::MPSGraphPaddingStyleTF_SAME
                         dataLayout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNHWC
        ];

        dst = [graph maxPooling2DWithSourceTensor: dst
                                       descriptor: descr
                                             name: nil];
      }
      else if (postOp == PostOp::Upsample)
      {
        dst = [graph resizeTensor: dst
                             size: @[@(dstDesc.getH()), @(dstDesc.getW())]
                             mode: MPSGraphResizeMode::MPSGraphResizeNearest
                     centerResult: true
                     alignCorners: false
                           layout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNHWC
                             name: nil];
      }
      else if (postOp != PostOp::None)
        throw std::invalid_argument("unsupported convolution postop");

      dstNode->tensor = dst;
    });

    return conv;
  }

  std::shared_ptr<Op> MetalGraph::addConcatConv(const std::string& name,
                                                const std::shared_ptr<Op>& src1Op,
                                                const std::shared_ptr<Op>& src2Op,
                                                Activation activation)
  {
    if (!inputProcess || outputProcess)
      throw std::logic_error("op must be added to the graph between input and output processing");

    TensorNode* src1Node = tensorNodesByOp[src1Op.get()];
    TensorNode* src2Node = tensorNodesByOp[src2Op.get()];

    TensorDesc src1Desc = src1Node->desc;
    TensorDesc src2Desc = src2Node->desc;

    TensorDims dstDims{src1Desc.getC() + src2Desc.getC(), src1Desc.getH(), src1Desc.getW()};
    TensorDesc dstDesc{dstDims, src1Desc.layout, src1Desc.dataType};

    auto concat = std::make_shared<MetalOp>();
    TensorNode* concatDstNode = addOp(concat, dstDesc);

    lazyInits.push_back([=]()
    {
      concatDstNode->tensor = [graph concatTensors: @[src1Node->tensor, src2Node->tensor]
                                         dimension: 3
                                              name: nil];
    });

    return addConv(name, concat, activation);
  }

  std::shared_ptr<Op> MetalGraph::addPool(const std::string& name,
                                          const std::shared_ptr<Op>& srcOp)
  {
    throw std::runtime_error("not implemented");
  }

  std::shared_ptr<Op> MetalGraph::addUpsample(const std::string& name,
                                              const std::shared_ptr<Op>& srcOp)
  {
    throw std::runtime_error("not implemented");
  }

  double MetalGraph::getWorkAmount() const
  {
    return 3; // input process, graph, output process
  }

  bool MetalGraph::isSupported() const
  {
    for (const auto& op : ops)
      if (!op->isSupported())
        return false;

    return true;
  }

  size_t MetalGraph::getScratchByteSize()
  {
    return scratchByteSize;
  }

  void MetalGraph::setScratch(const Ref<Buffer>& scratch)
  {
    if (scratch->getByteSize() < getScratchByteSize())
      throw std::invalid_argument("graph scratch buffer too small");
    this->scratch = scratch;
  }

  void MetalGraph::cleanup()
  {
    lazyInits.clear();
    tensorNodesByOp.clear();
    tensorNodes.clear();
  }

  void MetalGraph::clear()
  {
    if (finalized)
      throw std::runtime_error("graph cannot be cleared after finalization");

    cleanup();
    ops.clear();
    scratchByteSize = 0;
    privateByteSize = 0;
    dirty = false;
  }

  void MetalGraph::finalize()
  {
    if (!inputProcess || !outputProcess)
      throw std::logic_error("graph must have input and output processing");

    graph = [[MPSGraph alloc] init];

    graphExecDesc = [MPSGraphExecutionDescriptor new];
    graphExecDesc.completionHandler = ^(MPSGraphTensorDataDictionary* resultsDictionary,
                                        NSError* _Nullable error) {};

    for (auto& lazyInit : lazyInits)
      lazyInit();
    lazyInits.clear();

    for (auto& op : ops)
      op->finalize();

    cleanup();
    constTensors.reset();

    finalized = true;
  }

  void MetalGraph::run(Progress& progress)
  {
    if (!finalized)
      throw std::logic_error("graph not finalized");

    // Submit input processing
    inputProcess->submit();
    progress.update(engine, 1);

    // Submit graph
    MPSGraphTensorData* graphInputData  = newMPSGraphTensorData(inputProcess->getDst());
    MPSGraphTensorData* graphOutputData = newMPSGraphTensorData(outputProcess->getSrc());
    MPSCommandBuffer* commandBuffer = engine->getMPSCommandBuffer();

    [graph encodeToCommandBuffer: commandBuffer
                           feeds: @{graphInput: graphInputData}
                targetOperations: nil
               resultsDictionary: @{graphOutput: graphOutputData}
             executionDescriptor: graphExecDesc];

    [commandBuffer commit];

    [graphInputData release];
    [graphOutputData release];

    progress.update(engine, 1);

    // Submit output processing
    outputProcess->submit();
    progress.update(engine, 1);
  }

OIDN_NAMESPACE_END
