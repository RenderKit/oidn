// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor.h"
#include "image.h"
#include "node.h"
#include "input_reorder.h"
#include "output_reorder.h"
#include "output_copy.h"
#include "color.h"
#include "progress.h"

#pragma once

namespace oidn {

  class Executable
  {
  public:
    virtual ~Executable() {}
    virtual void execute(Progress& progress) = 0;
    virtual double getWorkAmount() const = 0; // for progress reporting
  };

  class Network : public Executable
  {
  public:
    Network(const Ref<Device>& device, const std::map<std::string, Ref<Tensor>>& weightsMap);

    void execute(Progress& progress) override;
    double getWorkAmount() const override;

    Ref<Tensor> newTensor(const TensorDims& dims);

    TensorDims getInputReorderDims(const TensorDims& srcDims, int alignment);

    std::shared_ptr<Node> addInputReorder(const Image& color,
                                          const Image& albedo,
                                          const Image& normal,
                                          const std::shared_ptr<TransferFunction>& transferFunc,
                                          bool hdr,
                                          int alignment,
                                          const Ref<Tensor>& dst);

    std::shared_ptr<Node> addOutputReorder(const Ref<Tensor>& src,
                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                           bool hdr,
                                           const Image& output);

    TensorDims getConvDims(const std::string& name, const TensorDims& srcDims);
    std::shared_ptr<Node> addConv(const std::string& name,
                                  const Ref<Tensor>& src,
                                  const Ref<Tensor>& dst,
                                  bool relu = true);

    TensorDims getPoolDims(const TensorDims& srcDims);
    std::shared_ptr<Node> addPool(const Ref<Tensor>& src,
                                  const Ref<Tensor>& dst);

    TensorDims getUpsampleDims(const TensorDims& srcDims);
    std::shared_ptr<Node> addUpsample(const Ref<Tensor>& src,
                                      const Ref<Tensor>& dst);

    TensorDims getConcatDims(const std::vector<TensorDims>& srcDims);
    std::vector<Ref<Tensor>> getConcatSrc(const Ref<Tensor>& dst, const std::vector<TensorDims>& srcDims);

    std::shared_ptr<Node> addAutoexposure(const Image& color,
                                          const std::shared_ptr<TransferFunction>& transferFunc);

    void finalize();

  private:
    Ref<Device> device;
    int K; // block size of blocked tensor layouts

    std::vector<std::shared_ptr<Node>> nodes;
    std::map<std::string, Ref<Tensor>> weightsMap;

    // Memory allocation statistics
    size_t activationAllocBytes = 0; // number of allocated activation bytes
    size_t totalAllocBytes      = 0; // total number of allocated bytes

    Ref<Tensor> padWeights(const Ref<Tensor>& src);
    Ref<Tensor> padBias(const Ref<Tensor>& src);
  };

} // namespace oidn
