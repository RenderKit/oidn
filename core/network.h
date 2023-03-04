// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_process.h"
#include "output_process.h"
#include "conv.h"
#include "concat_conv.h"
#include "pool.h"
#include "upsample.h"
#include "image_copy.h"
#include "progress.h"
#include "scratch.h"
#include "data.h"
#include <unordered_map>

OIDN_NAMESPACE_BEGIN

  // Neural network consisting of operations
  class Network
  {
  public:
    Network(const Ref<Engine>& engine, const Data& weightsBlob);

    std::shared_ptr<InputProcess> addInputProcess(const std::string& name,
                                                  const TensorDims& srcDims,
                                                  int alignment,
                                                  const std::shared_ptr<TransferFunction>& transferFunc,
                                                  bool hdr,
                                                  bool snorm);

    std::shared_ptr<OutputProcess> addOutputProcess(const std::string& name,
                                                    const TensorDesc& srcDesc,
                                                    const std::shared_ptr<TransferFunction>& transferFunc,
                                                    bool hdr,
                                                    bool snorm);

    std::shared_ptr<Conv> addConv(const std::string& name,
                                  const TensorDesc& srcDesc,
                                  Activation activation = Activation::ReLU,
                                  PostOp postOp = PostOp::None);

    std::shared_ptr<ConcatConv> addConcatConv(const std::string& name,
                                              const TensorDesc& src1Desc,
                                              const TensorDesc& src2Desc,
                                              Activation activation = Activation::ReLU);

    std::shared_ptr<Pool> addPool(const std::string& name,
                                  const TensorDesc& srcDesc);

    std::shared_ptr<Upsample> addUpsample(const std::string& name,
                                          const TensorDesc& srcDesc);

    bool isSupported() const;

    size_t getScratchAlignedSize() const;
    void setScratch(const std::shared_ptr<Tensor>& scratch);

    size_t getPrivateByteSize() const { return privateByteSize; }

    double getWorkAmount() const;
    void clear();
    void finalize();
    void run(Progress& progress);

  private:
    template<typename SrcT, typename DstT, TensorLayout srcLayout, TensorLayout dstLayout>
    bool tryReorderWeight(const Tensor& src, int srcBeginI, int srcI, Tensor& dst, int dstBeginI, int dstI);

    void reorderWeight(const Tensor& src, int srcBeginI, int srcI, Tensor& dst, int dstBeginI, int dstI);

    template<typename SrcT, typename DstT>
    bool tryReorderBias(const Tensor& src, Tensor& dst);

    void reorderBias(const Tensor& src, Tensor& dst);

    Ref<Engine> engine;
    std::vector<std::shared_ptr<Op>> ops;
    size_t privateByteSize = 0;

    // Used only while building the network
    std::vector<std::function<void()>> lazyInits; // lazy initialization for some ops
    std::unordered_map<std::string, std::shared_ptr<Tensor>> weights;
  };

OIDN_NAMESPACE_END
