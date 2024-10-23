// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_process.h"
#include "output_process.h"
#include "conv.h"
#include "concat_conv.h"
#include "pool.h"
#include "upsample.h"
#include "progress.h"
#include "arena_planner.h"
#include <vector>
#include <unordered_map>

OIDN_NAMESPACE_BEGIN

  class Graph final : public Op
  {
  public:
    Graph(Engine* engine,
          const std::shared_ptr<TensorMap>& constTensors,
          const std::shared_ptr<TensorMap>& cachedConstTensors,
          bool fastMath = false);

    Engine* getEngine() const override { return engine; }

    Ref<InputProcess> addInputProcess(const std::string& name,
                                      const TensorDims& srcDims,
                                      const std::shared_ptr<TransferFunction>& transferFunc,
                                      bool hdr,
                                      bool snorm);

    Ref<OutputProcess> addOutputProcess(const std::string& name,
                                        const Ref<Op>& srcOp,
                                        const std::shared_ptr<TransferFunction>& transferFunc,
                                        bool hdr,
                                        bool snorm);

    Ref<Op> addConv(const std::string& name,
                    const Ref<Op>& srcOp,
                    Activation activation,
                    PostOp postOp = PostOp::None);

    Ref<Op> addConcatConv(const std::string& name,
                          const Ref<Op>& src1Op,
                          const Ref<Op>& src2Op,
                          Activation activation);

    Ref<Op> addPool(const std::string& name,
                    const Ref<Op>& srcOp);

    Ref<Op> addUpsample(const std::string& name,
                        const Ref<Op>& srcOp);

    bool isSupported() const override;

    size_t getScratchByteSize() override;
    void setScratch(const Ref<Buffer>& scratch) override;
    size_t getPrivateByteSize() { return privateByteSize; }

    size_t getWorkAmount() const override { return workAmount; }
    void clear();
    void finalize() override;
    void submit(const Ref<Progress>& progress) override;

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

    void addOp(const Ref<Op>& op, const std::vector<Ref<Op>>& srcOps,
               bool concatSrcs = false);

    std::shared_ptr<TensorAlloc> addOp(const Ref<Op>& op,
                                       const std::vector<Ref<Op>>& srcOps,
                                       const TensorDesc& dstDesc,
                                       bool concatSrcs = false);

    void planAllocs();
    void cleanup();

    Ref<Tensor> getCachedConstTensor(const std::string& name, const TensorDesc& desc);
    void setCachedConstTensor(const std::string& name, const Ref<Tensor>& tensor);

    Engine* engine;
    std::vector<Ref<Op>> ops;
    Ref<Buffer> scratch;        // scratch buffer
    size_t scratchByteSize = 0; // total size of scratch data
    size_t privateByteSize = 0; // total size of private data (e.g. constant tensors)
    size_t workAmount = 0;      // total estimated amount of work for progress monitoring
    bool dirty = false;
    bool finalized = false;

    // Used only while building the graph
    ArenaPlanner tensorScratchPlanner;  // tensor scratch allocation planner
    size_t tensorScratchByteOffset = 0; // offset of tensor data in the scratch buffer
    std::unordered_map<Op*, std::shared_ptr<TensorAlloc>> tensorAllocs;
    std::vector<std::function<void()>> lazyInits;  // lazy initialization for ops
    std::shared_ptr<TensorMap> constTensors;       // original weights
    std::shared_ptr<TensorMap> cachedConstTensors; // cached final weights shared with other graphs
    bool fastMath = false;
  };

OIDN_NAMESPACE_END
