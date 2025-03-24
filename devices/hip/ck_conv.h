// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// FIXME: workaround for compile error when building for target unsupported by CK
#include "ck/ck.hpp"
#if !defined(CK_BUFFER_RESOURCE_3RD_DWORD)
  #define CK_BUFFER_RESOURCE_3RD_DWORD -1
#endif

#include "core/conv.h"
#include "hip_engine.h"
#include "ck/utility/data_type.hpp"

OIDN_NAMESPACE_BEGIN

  template<typename T>
  struct CKDataType { using Type = T; };

  template<>
  struct CKDataType<half> { using Type = ck::half_t; };

  template<ck::index_t... Is>
  using S = ck::Sequence<Is...>;

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

  struct CKConvFactory
  {
    Ref<Conv> (*make)(HIPEngine*, const ConvDesc&);

    DataType dataType;
    DataType accumType;
    Activation activation;
    int blockM, blockN, blockK; // threadblock size
  };

  template<HIPArch arch>
  std::vector<CKConvFactory> getCKConvInstances(Activation activation);

OIDN_NAMESPACE_END