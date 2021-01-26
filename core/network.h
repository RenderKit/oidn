// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include "tensor.h"
#include "image.h"
#include "node.h"
#include "color.h"
#include "progress.h"

#pragma once

namespace oidn {

  class Network : public RefCount
  {
  public:
    Network(const Ref<Device>& device, const std::map<std::string, Ref<Tensor>>& weightsMap);

    void execute(Progress& progress);
    double getWorkAmount() const;

    Ref<Tensor> newTensor(const TensorDims& dims);

    TensorDims getInputReorderDims(const TensorDims& srcDims, int alignment);

    Ref<Node> addInputReorder(const Image& color,
                              const Image& albedo,
                              const Image& normal,
                              const Ref<Tensor>& dst,
                              const Ref<TransferFunction>& transferFunc,
                              bool hdr,
                              bool snorm,
                              int alignment);

    Ref<Node> addOutputReorder(const Ref<Tensor>& src,
                               const Image& output,
                               const Ref<TransferFunction>& transferFunc,
                               bool hdr,
                               bool snorm);

    TensorDims getConvDims(const std::string& name, const TensorDims& srcDims);
    Ref<Node> addConv(const std::string& name,
                      const Ref<Tensor>& src,
                      const Ref<Tensor>& dst,
                      bool relu = true);

    TensorDims getPoolDims(const TensorDims& srcDims);
    Ref<Node> addPool(const Ref<Tensor>& src,
                      const Ref<Tensor>& dst);

    TensorDims getUpsampleDims(const TensorDims& srcDims);
    Ref<Node> addUpsample(const Ref<Tensor>& src,
                          const Ref<Tensor>& dst);

    TensorDims getConcatDims(const std::vector<TensorDims>& srcDims);
    std::vector<Ref<Tensor>> getConcatSrc(const Ref<Tensor>& dst,
                                          const std::vector<TensorDims>& srcDims);

    Ref<Node> addAutoexposure(const Image& color,
                              const Ref<TransferFunction>& transferFunc);

    void finalize();

  private:
    Ref<Device> device;
    int K; // block size of blocked tensor layouts

    std::vector<Ref<Node>> nodes;
    std::map<std::string, Ref<Tensor>> weightsMap;

    // Memory allocation statistics
    size_t activationAllocBytes = 0; // number of allocated activation bytes
    size_t totalAllocBytes      = 0; // total number of allocated bytes

    Ref<Tensor> padWeights(const Ref<Tensor>& src);
    Ref<Tensor> padBias(const Ref<Tensor>& src);
  };

} // namespace oidn
