// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "core/conv.h"
#include "hip_engine.h"
#include "ck/utility/data_type.hpp"

OIDN_NAMESPACE_BEGIN

  template<typename T>
  struct CKDataType { using Type = T; };

  template<>
  struct CKDataType<half> { using Type = ck::half_t; };

  inline std::array<ck::index_t, 5> getCKTensorLengths(const TensorDesc& td)
  {
    switch (td.layout)
    {
    case TensorLayout::x:
      return {1, 1, td.getPaddedX(), 1, 1}; // GNCHW
    case TensorLayout::hwc:
      return {1, 1, td.getPaddedC(), td.getH(), td.getW()}; // GNCHW
    case TensorLayout::ohwi:
      return {1, td.getPaddedO(), td.getPaddedI(), td.getH(), td.getW()}; // GKCYX
    default:
      throw std::invalid_argument("unsupported tensor layout");
    }
  }

  inline std::array<ck::index_t, 5> getCKTensorStrides(const TensorDesc& td)
  {
    switch (td.layout)
    {
    case TensorLayout::x:
      return {0, 0, 1, 0, 0}; // GNCHW
    case TensorLayout::hwc:
      return {0, 0, 1, td.getW() * td.getPaddedC(), td.getPaddedC()}; // GNCHW
    case TensorLayout::ohwi:
      return {0, td.getH() * td.getW() * td.getPaddedI(),
              1, td.getW() * td.getPaddedI(), td.getPaddedI()}; // GKCYX
    default:
      throw std::invalid_argument("unsupported tensor layout");
    }
  }

OIDN_NAMESPACE_END