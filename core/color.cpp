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

  TransferFunction::operator ispc::TransferFunction() const
  {
    ispc::TransferFunction res;

    switch (type)
    {
    case Type::Linear: ispc::LinearTransferFunction_Constructor(&res); break;
    case Type::SRGB:   ispc::SRGBTransferFunction_Constructor(&res);   break;
    case Type::PU:     ispc::PUTransferFunction_Constructor(&res);     break;
    case Type::Log:    ispc::LogTransferFunction_Constructor(&res);    break;
    default:
      assert(0);
    }

    res.inputScale  = getInputScale();
    res.outputScale = getOutputScale();
    
    return res;
  }

} // namespace oidn
