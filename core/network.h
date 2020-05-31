// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common/tensor.h"
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
    Network(const Ref<Device>& device, const std::map<std::string, Tensor>& weightsMap);

    void execute(Progress& progress) override;
    double getWorkAmount() const override;

    std::shared_ptr<memory> allocMemory(const memory::dims& dims,
                                        memory::format_tag format = memory::format_tag::any,
                                        void* data = nullptr);

    std::shared_ptr<memory> castMemory(const memory::dims& dims,
                                       const std::shared_ptr<memory>& src,
                                       size_t srcOffset = 0,
                                       memory::format_tag format = memory::format_tag::any);

    std::shared_ptr<memory> castMemory(const memory::dims& dims,
                                       const std::shared_ptr<memory>& src,
                                       const memory::dims& srcOffset);

    void zeroMemory(const std::shared_ptr<memory>& dst);

    memory::dims getInputReorderDims(const memory::dims& srcDims, int alignment);

    std::shared_ptr<Node> addInputReorder(const Image& color,
                                          const Image& albedo,
                                          const Image& normal,
                                          const std::shared_ptr<TransferFunction>& transferFunc,
                                          bool hdr,
                                          int alignment,
                                          const std::shared_ptr<memory>& userDst = nullptr);

    std::shared_ptr<Node> addOutputReorder(const std::shared_ptr<memory>& src,
                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                           bool hdr,
                                           const Image& output);

    memory::dims getConvDims(const std::string& name, const memory::dims& srcDims);
    std::shared_ptr<Node> addConv(const std::string& name,
                                  const std::shared_ptr<memory>& src,
                                  const std::shared_ptr<memory>& userDst = nullptr,
                                  bool relu = true);

    memory::dims getPoolDims(const memory::dims& srcDims);
    std::shared_ptr<Node> addPool(const std::shared_ptr<memory>& src,
                                  const std::shared_ptr<memory>& userDst = nullptr);

    memory::dims getUpsampleDims(const memory::dims& srcDims);
    std::shared_ptr<Node> addUpsample(const std::shared_ptr<memory>& src,
                                      const std::shared_ptr<memory>& userDst = nullptr);

    memory::dims getConcatDims(const memory::dims& src1Dims, const memory::dims& src2Dims);

    std::shared_ptr<Node> addAutoexposure(const Image& color,
                                          const std::shared_ptr<TransferFunction>& transferFunc);

    void finalize();

  private:
    Ref<Device> device;
    engine eng;
    stream sm;
    int K;                     // block size
    memory::format_tag nChwKc; // native blocked format

    std::vector<std::shared_ptr<Node>> nodes;
    std::map<std::string, Tensor> weightsMap;

    // Memory allocation statistics
    size_t activationAllocBytes = 0; // number of allocated activation bytes
    size_t totalAllocBytes      = 0; // total number of allocated bytes

    std::shared_ptr<memory> padWeights(const Tensor& src);
    std::shared_ptr<memory> padBias(const Tensor& src);
  };

} // namespace oidn
