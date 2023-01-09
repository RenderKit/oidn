// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "color.h"

namespace oidn {

  constexpr float TransferFunction::yMax;

  TransferFunction::TransferFunction(Type type)
    : type(type)
  {
    const float xMax = math::reduce_max(forward(yMax));
    normScale    = 1./xMax;
    rcpNormScale = xMax;
  }

} // namespace oidn
