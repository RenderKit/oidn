// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include "tensor.h"
#include "image.h"
#include "node.h"
#include "input_reorder.h"
#include "output_reorder.h"
#include "progress.h"
#include "scratch.h"

#pragma once

namespace oidn {

  class Network
  {
  public:
    Network(const Ref<Device>& device, const std::map<std::string, std::shared_ptr<Tensor>>& weightsMap);

    void execute(Progress& progress);
    double getWorkAmount() const;

    // Scratch memory
    void allocScratch(size_t size);
    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, ptrdiff_t offset);
    std::shared_ptr<Image> newImage(const ImageDesc& desc, ptrdiff_t offset);

    TensorDesc getInputReorderDesc(const TensorDims& srcDims, int alignment);

    std::shared_ptr<InputReorderNode> addInputReorder(const std::string& name,
                                                      const std::shared_ptr<Tensor>& dst,
                                                      const std::shared_ptr<TransferFunction>& transferFunc,
                                                      bool hdr,
                                                      bool snorm);

    std::shared_ptr<OutputReorderNode> addOutputReorder(const std::string& name,
                                                        const std::shared_ptr<Tensor>& src,
                                                        const std::shared_ptr<TransferFunction>& transferFunc,
                                                        bool hdr,
                                                        bool snorm);

    TensorDesc getConvDesc(const std::string& name, const TensorDesc& srcDesc);
    std::shared_ptr<Node> addConv(const std::string& name,
                                  const std::shared_ptr<Tensor>& src,
                                  const std::shared_ptr<Tensor>& dst,
                                  bool relu = true);

    TensorDesc getPoolDesc(const TensorDesc& srcDesc);
    std::shared_ptr<Node> addPool(const std::string& name,
                                  const std::shared_ptr<Tensor>& src,
                                  const std::shared_ptr<Tensor>& dst);

    TensorDesc getUpsampleDesc(const TensorDesc& srcDesc);
    std::shared_ptr<Node> addUpsample(const std::string& name,
                                      const std::shared_ptr<Tensor>& src,
                                      const std::shared_ptr<Tensor>& dst);

    TensorDesc getConcatDesc(const std::vector<TensorDesc>& srcDescs);

    void finalize();

  private:
    Ref<Device> device;
    int K; // block size of blocked tensor layouts

    std::vector<std::shared_ptr<Node>> nodes;
    std::map<std::string, std::shared_ptr<Tensor>> weightsMap;
    Ref<ScratchBuffer> scratch;

    std::shared_ptr<Tensor> padWeights(const std::shared_ptr<Tensor>& src);
    std::shared_ptr<Tensor> padBias(const std::shared_ptr<Tensor>& src);
  };

} // namespace oidn
