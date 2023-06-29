// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/graph.h"
#include "metal_engine.h"
#include "metal_common.h"
#include "metal_op.h"
#include <unordered_map>
#include <vector>

OIDN_NAMESPACE_BEGIN

  class MetalEngine;

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
    void cleanup();

    void addOp(std::shared_ptr<MetalOp>& op, TensorDesc td);

    MPSGraphTensor_t createConv(std::shared_ptr<MetalOp>& op, MPSGraphTensor_t input);
    MPSGraphTensor_t createActivation(std::shared_ptr<MetalOp>& op, MPSGraphTensor_t input);
    MPSGraphTensor_t createPool(std::shared_ptr<MetalOp>& op, MPSGraphTensor_t input);
    MPSGraphTensor_t createConcat(std::shared_ptr<MetalOp>& op,
                                  MPSGraphTensor_t input1, MPSGraphTensor_t input2);
    MPSGraphTensor_t createUpsample(std::shared_ptr<MetalOp>& op, MPSGraphTensor_t input);

  private:
    Ref<MetalEngine> engine;
    std::shared_ptr<TensorMap> constTensors;
    bool fastMath = false;

    std::vector<std::shared_ptr<MetalOp>> ops;
    std::shared_ptr<InputProcess> inputProcess;
    std::shared_ptr<OutputProcess> outputProcess;

    std::unordered_map<Op*, TensorDesc> tensorDescByOp;

    size_t opScratchByteSize     = 0; // total size of operation scratch
    size_t tensorScratchByteSize = 0; // total size of temporary tensors
    size_t constByteSize         = 0; // total size of constant tensors
    bool dirty = false;
    bool finalized = false;

    MPSGraph_t graph;

    Ref<Buffer> inputBuffer;

    MPSGraphTensor_t graphInput;
    MPSGraphTensor_t graphOutput;

    std::shared_ptr<Tensor> outputTensor;
  };

OIDN_NAMESPACE_END
