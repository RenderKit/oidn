// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "graph.h"
#include "arena_planner.h"
#include <vector>
#include <unordered_map>

OIDN_NAMESPACE_BEGIN

  // Generic graph implementation used by most devices
  class GenericGraph final : public Graph
  {
  public:
    GenericGraph(const Ref<Engine>& engine,
                 const std::shared_ptr<TensorMap>& constTensors,
                 bool fastMath = false);

    std::shared_ptr<InputProcess> addInputProcess(
                                    const std::string& name,
                                    const TensorDims& srcDims,
                                    int tileAlignment,
                                    const std::shared_ptr<TransferFunction>& transferFunc,
                                    bool hdr,
                                    bool snorm) override;

    std::shared_ptr<OutputProcess> addOutputProcess(
                                     const std::string& name,
                                     const std::shared_ptr<Op>& srcOp,
                                     const std::shared_ptr<TransferFunction>& transferFunc,
                                     bool hdr,
                                     bool snorm) override;

    std::shared_ptr<Op> addConv(const std::string& name,
                                const std::shared_ptr<Op>& srcOp,
                                Activation activation,
                                PostOp postOp) override;

    std::shared_ptr<Op> addConcatConv(const std::string& name,
                                      const std::shared_ptr<Op>& src1Op,
                                      const std::shared_ptr<Op>& src2Op,
                                      Activation activation) override;

    std::shared_ptr<Op> addPool(const std::string& name,
                                const std::shared_ptr<Op>& srcOp) override;

    std::shared_ptr<Op> addUpsample(const std::string& name,
                                    const std::shared_ptr<Op>& srcOp) override;

    bool isSupported() const override;

    size_t getScratchByteSize() override;
    void setScratch(const Ref<Buffer>& scratch) override;
    size_t getPrivateByteSize() override { return privateByteSize; }

    double getWorkAmount() const override;
    void clear() override;
    void finalize() override;
    void run(Progress& progress) override;

  private:
    // Temporary tensor allocation
    struct TensorAlloc
    {
      TensorDesc desc; // tensor descriptor
      int id;          // allocation ID used by the scratch planner

      // Set only when planning allocations
      Ref<Tensor> tensor;

      TensorAlloc(const TensorDesc& desc, int id)
        : desc(desc),
          id(id) {}
    };

    void addOp(const std::shared_ptr<Op>& op, const std::vector<std::shared_ptr<Op>>& srcOps,
               bool concatSrcs = false);

    std::shared_ptr<TensorAlloc> addOp(const std::shared_ptr<Op>& op,
                                       const std::vector<std::shared_ptr<Op>>& srcOps,
                                       const TensorDesc& dstDesc,
                                       bool concatSrcs = false);

    void planAllocs();
    void cleanup();

    Ref<Engine> engine;
    std::vector<std::shared_ptr<Op>> ops;
    Ref<Buffer> scratch;        // scratch buffer
    size_t scratchByteSize = 0; // total size of scratch data
    size_t privateByteSize = 0; // total size of private data (e.g. constant tensors)
    bool dirty = false;
    bool finalized = false;

    // Used only while building the graph
    ArenaPlanner tensorScratchPlanner; // tensor scratch allocation planner
    size_t tensorScratchByteOffset = 0; // offset of tensor data in the scratch buffer
    std::unordered_map<Op*, std::shared_ptr<TensorAlloc>> tensorAllocs;
    std::vector<std::function<void()>> lazyInits; // lazy initialization for ops
    std::shared_ptr<TensorMap> constTensors;
    bool fastMath = false;
  };

OIDN_NAMESPACE_END
