// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <miopen/miopen.h>
#include "../tensor.h"
#include "hip_engine.h"

namespace oidn {

  void checkError(miopenStatus_t status);

  // Convert data type to MIOpen equivalent
  miopenDataType_t toMIOpen(DataType dataType);

  // Converts tensor descriptor to MIOpen equivalent
  // Returns nullptr if tensor dimensions exceed limit supported by MIOpen
  miopenTensorDescriptor_t toMIOpen(const TensorDesc& td);

} // namespace oidn
