// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mkl-dnn/include/dnnl.hpp"
#include "../tensor.h"

namespace oidn {

  dnnl::memory::desc toDNNL(const TensorDesc& td);

  // Returns the internal DNNL memory structure of a DNNLTensor
  const dnnl::memory& getDNNL(const Tensor& tensor);

} // namespace oidn