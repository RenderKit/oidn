// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cutlass_conv.h"

namespace oidn {

// Ampere
template<>
std::vector<CutlassConvFactory> getCutlassConvInstances<80>()
{
  using namespace cutlass::arch;
  using cutlass::gemm::GemmShape;

  return {
    CutlassConvInstance<half, Sm80, GemmShape<256, 32, 32>, GemmShape<64, 32, 32>, 3 /*4*/>::get(),
    CutlassConvInstance<half, Sm80, GemmShape<256, 64, 32>, GemmShape<64, 64, 32>, 3>::get(),
  };
}

} // namespace oidn