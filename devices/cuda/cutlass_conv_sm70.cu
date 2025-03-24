// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cutlass_conv.h"

OIDN_NAMESPACE_BEGIN

// Volta (SM 7.0)
template<>
std::vector<CutlassConvFactory> getCutlassConvInstances<70>()
{
  using namespace cutlass::arch;
  using cutlass::gemm::GemmShape;

  return
  {
    CutlassConvInstance<half, float, Sm70, GemmShape<256, 32, 32>, GemmShape<64, 32, 32>, 2>::get(),
    CutlassConvInstance<half, float, Sm70, GemmShape<256, 64, 32>, GemmShape<64, 64, 32>, 2>::get(),

    CutlassConvInstance<half, half,  Sm70, GemmShape<256, 32, 32>, GemmShape<64, 32, 32>, 2>::get(),
    CutlassConvInstance<half, half,  Sm70, GemmShape<256, 64, 32>, GemmShape<64, 64, 32>, 2>::get(),
  };
}

OIDN_NAMESPACE_END