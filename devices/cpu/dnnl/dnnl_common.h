// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/tensor.h"
#include "dnnl_engine.h"

OIDN_NAMESPACE_BEGIN

  dnnl::memory::data_type toDNNL(DataType dataType);
  dnnl::memory::desc toDNNL(const TensorDesc& td);

  // Creates a DNNL memory structure for a buffer
  dnnl::memory toDNNL(const Ref<Buffer>& buffer);

  // Returns the internal DNNL memory structure of a DNNLTensor
  const dnnl::memory& getDNNL(const Ref<Tensor>& tensor);

OIDN_NAMESPACE_END