// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor.h"

OIDN_NAMESPACE_BEGIN

  void reorderWeight(Tensor& src, int srcBeginI, int srcI, Tensor& dst, int dstBeginI, int dstI);
  void reorderBias(Tensor& src, Tensor& dst);

OIDN_NAMESPACE_END
