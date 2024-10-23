// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_pool.h"

OIDN_NAMESPACE_BEGIN

  BNNSPool::BNNSPool(BNNSEngine* engine, const PoolDesc& desc)
    : Pool(desc),
      engine(engine)
  {}

  BNNSPool::~BNNSPool()
  {
    if (filter)
      BNNSFilterDestroy(filter);
  }

  void BNNSPool::finalize()
  {
    if (filter)
      throw std::logic_error("pooling already finalized");

    BNNSLayerParametersPooling params = {
      .i_desc = toBNNS(srcDesc),
      .o_desc = toBNNS(dstDesc),
      .pooling_function = BNNSPoolingFunctionMax,
      .k_width  = 2,
      .k_height = 2,
      .x_stride = 2,
      .y_stride = 2
    };

    filter = BNNSFilterCreateLayerPooling(&params, nullptr);
    if (!filter)
      throw std::runtime_error("BNNSFilterCreateLayerPooling failed");
  }

  void BNNSPool::submitKernels(const Ref<CancellationToken>& ct)
  {
    if (!filter)
      throw std::logic_error("pooling not finalized");
    if (!src || !dst)
      throw std::logic_error("pooling source/destination not set");

    void* srcPtr = src->getPtr();
    void* dstPtr = dst->getPtr();
    engine->submitFunc([=] { BNNSFilterApply(filter, srcPtr, dstPtr); }, ct);
  }

OIDN_NAMESPACE_END