// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "color.h"

OIDN_NAMESPACE_BEGIN

  constexpr float TransferFunction::yMax;

  TransferFunction::TransferFunction(Type type)
    : type(type)
  {
    const float xMax = math::reduce_max(forward(yMax));
    normScale    = 1./xMax;
    rcpNormScale = xMax;
  }

OIDN_NAMESPACE_END
