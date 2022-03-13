// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mkl-dnn/include/dnnl.hpp"
#include "../tensor.h"

namespace oidn {

  dnnl::memory::data_type toDNNL(DataType dataType);
  dnnl::memory::desc toDNNL(const TensorDesc& td);

  // Returns the internal DNNL memory structure of a DNNLTensor
  const dnnl::memory& getDNNL(const std::shared_ptr<Tensor>& tensor);

} // namespace oidn