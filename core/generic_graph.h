// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "graph.h"
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
                                PostOp postOp) override;

    std::shared_ptr<ConcatConv> addConcatConv(const std::string& name,
                                              const std::shared_ptr<Op>& src1Op,
                                              const std::shared_ptr<Op>& src2Op,
                                              Activation activation) override;

    std::shared_ptr<Pool> addPool(const std::string& name,
                                  const std::shared_ptr<Op>& srcOp) override;

    std::shared_ptr<Upsample> addUpsample(const std::string& name,
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
    // Temporary tensor allocation record
    struct TensorAlloc
    {
      TensorDesc desc;   // tensor descriptor
      size_t byteSize;   // aligned size of the tensor
      int firstOpID;     // index of the first operation that uses this tensor
      int lastOpID;      // index of the last operation that uses this tensor
      TensorAlloc* next; // tensor allocated consecutively after this one
      TensorAlloc* prev; // tensor allocated consecutively before this one

      // Later set while/after planning allocations
      size_t byteOffset;
      std::shared_ptr<Tensor> tensor;

      TensorAlloc(const TensorDesc& desc, int firstOpID)
        : desc(desc),
          byteSize(desc.getAlignedSize()),
          firstOpID(firstOpID),
          lastOpID(firstOpID),
          next(nullptr),
          prev(nullptr),
          byteOffset(0) {}
    };

    void addOp(const std::shared_ptr<Op>& op, const std::vector<std::shared_ptr<Op>>& srcOps,
               bool concatSrcs = false);

    TensorAlloc* addOp(const std::shared_ptr<Op>& op, const std::vector<std::shared_ptr<Op>>& srcOps,
                       const TensorDesc& dstDesc, bool concatSrcs = false);

    void planAllocations();
    void cleanup();

    Ref<Engine> engine;
    std::vector<std::shared_ptr<Op>> ops;
    Ref<Buffer> scratch;
    size_t opScratchByteSize     = 0; // total size of operation scratch
    size_t tensorScratchByteSize = 0; // total size of temporary tensors
    size_t constByteSize         = 0; // total size of constant tensors
    bool dirty = false;
    bool finalized = false;

    // Used only while building the graph
    std::vector<std::unique_ptr<TensorAlloc>> tensorAllocs;
    std::unordered_map<Op*, TensorAlloc*> tensorAllocsByOp;
    std::vector<std::function<void()>> lazyInits; // lazy initialization for ops
    std::shared_ptr<TensorMap> constTensors;
    bool fastMath = false;
  };

OIDN_NAMESPACE_END
