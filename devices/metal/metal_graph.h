// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/graph.h"
#include "metal_engine.h"
#include "metal_common.h"
#include <unordered_map>
#include <vector>

OIDN_NAMESPACE_BEGIN

  class MetalEngine;

  class MetalOp final : public Op
  {
  public:
    void submit() override { throw std::logic_error("MetalOp cannot be submitted"); }
  };

  class MetalConv final : public Conv
  {
  public:
    explicit MetalConv(const ConvDesc& desc) : Conv(desc) {}
    void submit() override { throw std::logic_error("MetalConv cannot be submitted"); }
  };

  class MetalGraph final : public Graph
  {
  public:
    MetalGraph(const Ref<MetalEngine>& engine,
               const std::shared_ptr<TensorMap>& constTensors,
               bool fastMath = false);

    ~MetalGraph();

    std::shared_ptr<InputProcess> addInputProcess(const std::string& name,
                                                  const TensorDims& srcDims,
                                                  int tileAlignment,
                                                  const std::shared_ptr<TransferFunction>& transferFunc,
                                                  bool hdr,
                                                  bool snorm) override;

    std::shared_ptr<OutputProcess> addOutputProcess(const std::string& name,
                                                    const std::shared_ptr<Op>& srcOp,
                                                    const std::shared_ptr<TransferFunction>& transferFunc,
                                                    bool hdr,
                                                    bool snorm) override;

    std::shared_ptr<Op> addConv(const std::string& name,
                                const std::shared_ptr<Op>& srcOp,
                                Activation activation,
                                PostOp postOp = PostOp::None) override;

    std::shared_ptr<Op> addConcatConv(const std::string& name,
                                      const std::shared_ptr<Op>& src1Op,
                                      const std::shared_ptr<Op>& src2Op,
                                      Activation activation) override;

    std::shared_ptr<Op> addPool(const std::string& name,
                                const std::shared_ptr<Op>& srcOp) override;

    std::shared_ptr<Op> addUpsample(const std::string& name,
                                    const std::shared_ptr<Op>& srcOp) override;

    bool isSupported() const override;

    size_t getScratchAlignedSize() override;
    void setScratch(const Ref<Buffer>& scratch) override;
    size_t getPrivateByteSize() const override { return constByteSize; }

    double getWorkAmount() const override;
    void clear() override;
    void finalize() override;
    void run(Progress& progress) override;

  private:
    struct TensorNode
    {
      TensorDesc desc;
      MPSGraphTensor* tensor;
    };

    TensorNode* addOp(const std::shared_ptr<Op>& op, const TensorDesc& dstDesc);
    void cleanup();

    Ref<MetalEngine> engine;
    std::vector<std::shared_ptr<Op>> ops;
    std::shared_ptr<InputProcess> inputProcess;
    std::shared_ptr<OutputProcess> outputProcess;

    MPSGraph* graph = nullptr;
    MPSGraphTensor* graphInput = nullptr;
    MPSGraphTensor* graphOutput = nullptr;

    Ref<Buffer> inputBuffer;
    std::shared_ptr<Tensor> outputTensor;

    size_t opScratchByteSize     = 0; // total size of operation scratch
    size_t tensorScratchByteSize = 0; // total size of temporary tensors
    size_t constByteSize         = 0; // total size of constant tensors
    bool dirty = false;
    bool finalized = false;

    // Used only while building the graph
    std::vector<std::unique_ptr<TensorNode>> tensorNodes;
    std::unordered_map<Op*, TensorNode*> tensorNodesByOp;
    std::vector<std::function<void()>> lazyInits; // lazy initialization for ops
    std::shared_ptr<TensorMap> constTensors;
    bool fastMath = false;
  };

OIDN_NAMESPACE_END
