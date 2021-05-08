// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include "tensor.h"
#include "image.h"
#include "node.h"
#include "input_reorder.h"
#include "output_reorder.h"
#include "progress.h"

#pragma once

namespace oidn {

  class Network : public RefCount
  {
  public:
    Network(const Ref<Device>& device, const std::map<std::string, Ref<Tensor>>& weightsMap);

    void execute(Progress& progress);
    double getWorkAmount() const;

    // Scratch memory
    void setScratchSize(size_t size);
    Ref<Tensor> newTensor(const TensorDesc& desc, ptrdiff_t offset);
    Image newImage(const ImageDesc& desc, ptrdiff_t offset);

    TensorDesc getInputReorderDesc(const TensorDims& srcDims, int alignment);

    Ref<InputReorderNode> addInputReorder(const Ref<Tensor>& dst,
                                          const Ref<TransferFunction>& transferFunc,
                                          bool hdr,
                                          bool snorm);

    Ref<OutputReorderNode> addOutputReorder(const Ref<Tensor>& src,
                                            const Ref<TransferFunction>& transferFunc,
                                            bool hdr,
                                            bool snorm);

    TensorDesc getConvDesc(const std::string& name, const TensorDesc& srcDesc);
    Ref<Node> addConv(const std::string& name,
                      const Ref<Tensor>& src,
                      const Ref<Tensor>& dst,
                      bool relu = true);

    TensorDesc getPoolDesc(const TensorDesc& srcDesc);
    Ref<Node> addPool(const Ref<Tensor>& src,
                      const Ref<Tensor>& dst);

    TensorDesc getUpsampleDesc(const TensorDesc& srcDesc);
    Ref<Node> addUpsample(const Ref<Tensor>& src,
                          const Ref<Tensor>& dst);

    TensorDesc getConcatDesc(const std::vector<TensorDesc>& srcDescs);

    void finalize();

  private:
    Ref<Device> device;
    int K; // block size of blocked tensor layouts

    std::vector<Ref<Node>> nodes;
    std::map<std::string, Ref<Tensor>> weightsMap;

    Ref<Buffer> scratch;
    size_t scratchSize = 0;

    Ref<Tensor> padWeights(const Ref<Tensor>& src);
    Ref<Tensor> padBias(const Ref<Tensor>& src);
  };

} // namespace oidn
