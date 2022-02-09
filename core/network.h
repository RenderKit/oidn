// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <map>
#include "tensor.h"
#include "image.h"
#include "op.h"
#include "conv.h"
#include "concat_conv.h"
#include "pool.h"
#include "upsample.h"
#include "input_process.h"
#include "output_process.h"
#include "progress.h"
#include "scratch.h"
#include "weights.h"

#pragma once

namespace oidn {

  class Network
  {
  public:
    Network(const Ref<Device>& device, const std::shared_ptr<Weights>& weights);

    void run(Progress& progress);
    double getWorkAmount() const;

    // Scratch memory
    void allocScratch(size_t size);
    std::shared_ptr<Tensor> newTensor(const TensorDesc& desc, ptrdiff_t offset);
    std::shared_ptr<Image> newImage(const ImageDesc& desc, ptrdiff_t offset);

    TensorDesc getInputDesc(const TensorDims& srcDims, int alignment);

    std::shared_ptr<InputProcess> addInputProcess(const std::string& name,
                                                  const std::shared_ptr<Tensor>& dst,
                                                  const std::shared_ptr<TransferFunction>& transferFunc,
                                                  bool hdr,
                                                  bool snorm);

    std::shared_ptr<OutputProcess> addOutputProcess(const std::string& name,
                                                    const std::shared_ptr<Tensor>& src,
                                                    const std::shared_ptr<TransferFunction>& transferFunc,
                                                    bool hdr,
                                                    bool snorm);

    TensorDesc getConvDesc(const std::string& name, const TensorDesc& srcDesc);
    std::shared_ptr<Conv> addConv(const std::string& name,
                                  const std::shared_ptr<Tensor>& src,
                                  const std::shared_ptr<Tensor>& dst,
                                  bool relu = true);

    std::shared_ptr<Op> addConcatConv(const std::string& name,
                                      const std::shared_ptr<Tensor>& src1,
                                      const std::shared_ptr<Tensor>& src2,
                                      const std::shared_ptr<Tensor>& dst,
                                      bool relu = true);

    TensorDesc getPoolDesc(const TensorDesc& srcDesc);
    std::shared_ptr<Pool> addPool(const std::string& name,
                                  const std::shared_ptr<Tensor>& src,
                                  const std::shared_ptr<Tensor>& dst);

    TensorDesc getUpsampleDesc(const TensorDesc& srcDesc);
    std::shared_ptr<Upsample> addUpsample(const std::string& name,
                                          const std::shared_ptr<Tensor>& src,
                                          const std::shared_ptr<Tensor>& dst);

    TensorDesc getConcatDesc(const std::vector<TensorDesc>& srcDescs);

    void finalize();

  private:
    Ref<Device> device;
    int blockSize; // block size of blocked tensor layouts

    std::vector<std::shared_ptr<Op>> ops;
    std::shared_ptr<Weights> weights;
    Ref<ScratchBuffer> scratch;
  };

} // namespace oidn
