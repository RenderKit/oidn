// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "input_process.h"
#include "output_process.h"
#include "conv.h"
#include "concat_conv.h"
#include "pool.h"
#include "upsample.h"
#include "progress.h"

OIDN_NAMESPACE_BEGIN

  // Abstract graph of operations
  class Graph
  {
  public:
    virtual std::shared_ptr<InputProcess> addInputProcess(
                                            const std::string& name,
                                            const TensorDims& srcDims,
                                            int tileAlignment,
                                            const std::shared_ptr<TransferFunction>& transferFunc,
                                            bool hdr,
                                            bool snorm) = 0;

    virtual std::shared_ptr<OutputProcess> addOutputProcess(
                                             const std::string& name,
                                             const std::shared_ptr<Op>& srcOp,
                                             const std::shared_ptr<TransferFunction>& transferFunc,
                                             bool hdr,
                                             bool snorm) = 0;

    virtual std::shared_ptr<Op> addConv(const std::string& name,
                                        const std::shared_ptr<Op>& srcOp,
                                        Activation activation,
                                        PostOp postOp = PostOp::None) = 0;

    virtual std::shared_ptr<Op> addConcatConv(const std::string& name,
                                              const std::shared_ptr<Op>& src1Op,
                                              const std::shared_ptr<Op>& src2Op,
                                              Activation activation) = 0;

    virtual std::shared_ptr<Op> addPool(const std::string& name,
                                        const std::shared_ptr<Op>& srcOp) = 0;

    virtual std::shared_ptr<Op> addUpsample(const std::string& name,
                                            const std::shared_ptr<Op>& srcOp) = 0;

    virtual bool isSupported() const = 0;

    virtual size_t getScratchByteSize() = 0;
    virtual void setScratch(const Ref<Buffer>& scratch) = 0;
    virtual size_t getPrivateByteSize() = 0;

    virtual double getWorkAmount() const = 0;
    virtual void clear() = 0;
    virtual void finalize() = 0;
    virtual void run(Progress& progress) = 0;
  };

OIDN_NAMESPACE_END
