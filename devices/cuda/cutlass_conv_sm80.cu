// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cutlass_conv.h"

OIDN_NAMESPACE_BEGIN

// Ampere (SM 8.0), Ada Lovelace (SM 8.9), Hopper (SM 9.0), Blackwell (SM 10.0, 12.0)
template<>
std::vector<CutlassConvFactory> getCutlassConvInstances<80>()
{
  using namespace cutlass::arch;
  using cutlass::gemm::GemmShape;

  return
  {
    CutlassConvInstance<half, float, Sm80, GemmShape<256, 32, 32>, GemmShape<64, 32, 32>, 3 /*4*/>::get(),
    CutlassConvInstance<half, float, Sm80, GemmShape<256, 64, 32>, GemmShape<64, 64, 32>, 3>::get(),

    CutlassConvInstance<half, half,  Sm80, GemmShape<256, 32, 32>, GemmShape<64, 32, 32>, 3 /*4*/>::get(),
    CutlassConvInstance<half, half,  Sm80, GemmShape<256, 64, 32>, GemmShape<64, 64, 32>, 3>::get(),
  };
}

OIDN_NAMESPACE_END