// Copyright 2023 Apple Inc.
// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "metal_graph.h"
#include "metal_common.h"
#include "core/tensor_reorder.h"

OIDN_NAMESPACE_BEGIN

  MetalGraph::MetalGraph(const Ref<MetalEngine>& engine,
                         const std::shared_ptr<TensorMap>& constTensors,
                         bool fastMath)
    : engine(engine),
      constTensors(constTensors),
      fastMath(fastMath),
      graph(nullptr),
      commandQueue(nullptr) {}

  MetalGraph::~MetalGraph()
  {
    cleanup();
  }

  void MetalGraph::cleanup()
  {
    if (graph)
      [graph release];

    if (commandQueue)
      [commandQueue release];

    if (graphInput)
      [graphInput release];

    if (graphOutput)
      [graphOutput release];

    graph = nullptr;
    commandQueue = nullptr;
    graphInput = nullptr;
    graphOutput = nullptr;
  }

  std::shared_ptr<InputProcess> MetalGraph::addInputProcess(const std::string& name,
                                                            const TensorDims& srcDims,
                                                            int tileAlignment,
                                                            const std::shared_ptr<TransferFunction>& transferFunc,
                                                            bool hdr,
                                                            bool snorm)
  {
    inputProcess = engine->newInputProcess({srcDims, tileAlignment, transferFunc, hdr, snorm});
    inputProcess->setName(name);
    tensorDescByOp[inputProcess.get()] = inputProcess->getDstDesc();
    return inputProcess;
  }

  std::shared_ptr<OutputProcess> MetalGraph::addOutputProcess(const std::string& name,
                                                              const std::shared_ptr<Op>& srcOp,
                                                              const std::shared_ptr<TransferFunction>& transferFunc,
                                                              bool hdr,
                                                              bool snorm)
  {
    const TensorDesc srcDesc = tensorDescByOp[srcOp.get()];
    outputProcess = engine->newOutputProcess({srcDesc, transferFunc, hdr, snorm});
    outputProcess->setName(name);
    return outputProcess;
  }

  std::shared_ptr<Op> MetalGraph::addConv(const std::string& name,
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

    std::vector<std::shared_ptr<Op>> srcs = { srcOp };
    auto conv = std::make_shared<MetalOp>(MetalOpType::Conv, srcs);
    conv->setName(name);

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

    auto srcDesc = tensorDescByOp[srcOp.get()];

    TensorDims dstDims = {finalWeightDesc.getO(), srcDesc.getH(), srcDesc.getW()};
    auto dstDesc = TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
    addOp(conv, dstDesc);

    if (activation == Activation::ReLU)
    {
      std::vector<std::shared_ptr<Op>> srcs = { conv };
      auto relu = std::make_shared<MetalOp>(MetalOpType::Relu, srcs);
      relu->setName(name + "_relu");
      addOp(relu, dstDesc);
      return relu;
    }

    return conv;
  }

  std::shared_ptr<Op> MetalGraph::addConcatConv(const std::string& name,
                                                const std::shared_ptr<Op>& src1Op,
                                                const std::shared_ptr<Op>& src2Op,
                                                Activation activation)
  {
    std::vector<std::shared_ptr<Op>> srcs = { src1Op, src2Op };
    auto concat = std::make_shared<MetalOp>(MetalOpType::Concat, srcs);
    concat->setName(name + "_concat");

    auto src1Desc = tensorDescByOp[src1Op.get()];
    auto src2Desc = tensorDescByOp[src2Op.get()];

    auto dstDims = {src1Desc.getC() + src2Desc.getC(), src1Desc.getH(), src1Desc.getW()};
    auto dstDesc = TensorDesc(dstDims, src1Desc.layout, src1Desc.dataType);
    addOp(concat, dstDesc);

    return addConv(name, concat, activation);
  }

  std::shared_ptr<Op> MetalGraph::addPool(const std::string& name,
                                          const std::shared_ptr<Op>& srcOp)
  {
    std::vector<std::shared_ptr<Op>> srcs = { srcOp };
    auto pool = std::make_shared<MetalOp>(MetalOpType::Pool, srcs);
    pool->setName(name);

    auto srcDesc = tensorDescByOp[srcOp.get()];

    auto dstDims = {srcDesc.getC(), srcDesc.getH() / 2, srcDesc.getW() / 2};
    auto dstDesc = TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
    addOp(pool, dstDesc);

    return pool;
  }

  std::shared_ptr<Op> MetalGraph::addUpsample(const std::string& name,
                                              const std::shared_ptr<Op>& srcOp)
  {
    std::vector<std::shared_ptr<Op>> srcs = { srcOp };
    auto upsample = std::make_shared<MetalOp>(MetalOpType::Upsample, srcs);
    upsample->setName(name);

    auto srcDesc = tensorDescByOp[srcOp.get()];

    TensorDims dstDims = {srcDesc.getC(), srcDesc.getH() * 2, srcDesc.getW() * 2};
    auto dstDesc = TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
    addOp(upsample, dstDesc);

    return upsample;
  }

  double MetalGraph::getWorkAmount() const
  {
    return 3;
  }

  bool MetalGraph::isSupported() const
  {
    for (const auto& op : ops)
      if (!op->isSupported())
        return false;

    return true;
  }

  size_t MetalGraph::getScratchAlignedSize()
  {
    return opScratchByteSize + tensorScratchByteSize;
  }

  void MetalGraph::setScratch(const Ref<Buffer>& scratch)
  {
  }

  void MetalGraph::clear()
  {
    ops.clear();
    opScratchByteSize = 0;
    tensorScratchByteSize = 0;
    constByteSize = 0;
    dirty = false;
  }

  void MetalGraph::finalize()
  {
    MTLDevice_t device = static_cast<MetalDevice*>(engine->getDevice())->getMetalDevice();

    graph = [[MPSGraph alloc] init];
    commandQueue = [device newCommandQueue];

    std::unordered_map<Op*, MPSGraphTensor_t> tensorByOp;

    inputBuffer = engine->newBuffer(inputProcess->getDstDesc().getByteSize(), Storage::Device);

    inputProcess->finalize();

    inputProcess->setDst(engine->newTensor(inputBuffer, inputProcess->getDstDesc()));

    graphInput = toMPSGraphPlaceholder(graph, inputProcess->getDst()->getDesc());
    tensorByOp[inputProcess.get()] = graphInput;

    for (auto op : ops)
    {
      auto srcs = op->getSrc();
      auto input = tensorByOp[srcs[0].get()];

      MPSGraphTensor_t output = nullptr;

      switch (op->getOpType())
      {
        case MetalOpType::Conv:
          output = createConv(op, input);
          break;
        case MetalOpType::Relu:
          output = createActivation(op, input);
          break;
        case MetalOpType::Pool:
          output = createPool(op, input);
          break;
        case MetalOpType::Concat:
          output = createConcat(op, input, tensorByOp[srcs[1].get()]);
          break;
        case MetalOpType::Upsample:
          output = createUpsample(op, input);
          break;
        default:
          throw std::logic_error("not implemented");
      }

      tensorByOp[op.get()] = output;
    }

    auto lastOp = ops[ops.size() - 1].get();
    auto dstDesc = tensorDescByOp[lastOp];

    graphOutput = tensorByOp[lastOp];

    outputTensor = engine->newTensor(dstDesc);
    outputProcess->setSrc(outputTensor);

    outputProcess->finalize();

    constTensors.reset();
    finalized = true;
  }

  void MetalGraph::run(Progress& progress)
  {
    if (!finalized)
      throw std::logic_error("graph should be finalized first");

    auto lastOp = ops[ops.size() - 1].get();
    auto dstDesc = tensorDescByOp[lastOp];

    id<MTLBuffer> inputBuffer = getMTLBuffer(inputProcess->getDst()->getBuffer());
    id<MTLBuffer> outputBuffer = getMTLBuffer(outputTensor->getBuffer());

    MPSGraphTensorData_t graphInputData = toMPSGraphTensorData(inputBuffer, inputProcess->getDst());
    MPSGraphTensorData_t graphOutputData = toMPSGraphTensorData(outputBuffer, dstDesc);

    inputProcess->submit();
    progress.update(engine, 1);

    auto feeds = @{graphInput: graphInputData};
    auto results = @{graphOutput: graphOutputData};

    [graph runWithMTLCommandQueue: commandQueue
                            feeds: feeds
                 targetOperations: nil
                resultsDictionary: results];
    progress.update(engine, 1);

    outputProcess->submit();
    progress.update(engine, 1);

    [graphInputData release];
    [graphOutputData release];
    [feeds release];
    [results release];
  }

  void MetalGraph::addOp(std::shared_ptr<MetalOp>& op, TensorDesc td)
  {
    if (finalized)
      throw std::logic_error("graph cannot be changed after finalization");
    tensorDescByOp[op.get()] = td;
    ops.emplace_back(op);
  }

  MPSGraphTensor_t MetalGraph::createConv(std::shared_ptr<MetalOp>& op, MPSGraphTensor_t input)
  {
    auto name = op->getName();

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

    // Reorder the weight tensor
    auto finalWeight = engine->newTensor(finalWeightDesc, Storage::Host);
    reorderWeight(*weight, 0, weight->getI(),
                  *finalWeight, 0, finalWeight->getPaddedI());

    // Reorder the bias tensor
    auto finalBias = engine->newTensor(finalBiasDesc, Storage::Host);
    reorderBias(*bias, *finalBias);

    auto weightsTensor = toMPSGraphTensor(graph, finalWeight);
    auto biasTensor    = toMPSGraphTensor(graph, finalBias);

    auto outputTensor = [graph convolution2DWithSourceTensor: input
                                               weightsTensor: weightsTensor
                                                  descriptor: MPSGraphConvDesc()
                                                        name: nil];

    outputTensor = [graph additionWithPrimaryTensor: outputTensor
                                    secondaryTensor: biasTensor
                                               name: nil];

    constByteSize += finalWeightDesc.getByteSize() + finalBiasDesc.getByteSize();

    return outputTensor;
  }

  MPSGraphTensor_t MetalGraph::createActivation(std::shared_ptr<MetalOp>& op, MPSGraphTensor_t input)
  {
    return [graph reLUWithTensor: input
                            name: nil];
  }

  MPSGraphTensor_t MetalGraph::createPool(std::shared_ptr<MetalOp>& op, MPSGraphTensor_t input)
  {
    return [graph maxPooling2DWithSourceTensor: input
                                    descriptor: MPSGraphPoolDesc()
                                          name: nil];
  }

  MPSGraphTensor_t MetalGraph::createConcat(std::shared_ptr<MetalOp>& op,
                                            MPSGraphTensor_t input1, MPSGraphTensor_t input2)
  {
    id tensors = [NSMutableArray new];
    [tensors addObject: input1];
    [tensors addObject: input2];

    return [graph concatTensors: tensors
                      dimension: 3
                           name: nil];
  }

  MPSGraphTensor_t MetalGraph::createUpsample(std::shared_ptr<MetalOp>& op, MPSGraphTensor_t input)
  {
    MPSShape* shape = [input shape];
    MPSShape* size = @[[NSNumber numberWithInt: [shape[1] intValue] * 2],
                       [NSNumber numberWithInt: [shape[2] intValue] * 2]];

    return [graph resizeTensor: input
                          size: size
                          mode: MPSGraphResizeMode::MPSGraphResizeNearest
                  centerResult: true
                  alignCorners: false
                        layout: MPSGraphTensorNamedDataLayout::MPSGraphTensorNamedDataLayoutNHWC
                          name: nil];
  }

OIDN_NAMESPACE_END
