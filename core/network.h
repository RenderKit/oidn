// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_process.h"
#include "output_process.h"
#include "conv.h"
#include "concat_conv.h"
#include "pool.h"
#include "upsample.h"
#include "image_copy.h"
#include "progress.h"
#include "weights.h"
#include "scratch.h"

namespace oidn {

  // Neural network consisting of operations
  class Network
  {
  public:
    Network(const Ref<Device>& device, const std::shared_ptr<Weights>& weights);

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
                                  bool relu = true);

    std::shared_ptr<ConcatConv> addConcatConv(const std::string& name,
                                              const TensorDesc& src1Desc,
                                              const TensorDesc& src2Desc,
                                              bool relu = true);

    std::shared_ptr<Pool> addPool(const std::string& name,
                                  const TensorDesc& srcDesc);

    std::shared_ptr<Upsample> addUpsample(const std::string& name,
                                          const TensorDesc& srcDesc);

    const std::shared_ptr<Weights>& getWeights() const { return weights; }

    bool isSupported() const;

    size_t getScratchByteSize() const;
    void setScratch(const std::shared_ptr<Tensor>& scratch);

    double getWorkAmount() const;
    void clear();
    void finalize();
    void run(Progress& progress);

  private:
    Ref<Device> device;
    std::vector<std::shared_ptr<Op>> ops;
    std::shared_ptr<Weights> weights;
  };

} // namespace oidn
