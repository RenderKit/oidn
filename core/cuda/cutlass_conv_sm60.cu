// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cutlass_conv.h"

namespace oidn {

// Pascal
template<>
std::vector<CutlassConvFactory> getCutlassConvInstances<60>()
{
  using namespace cutlass::arch;
  using cutlass::gemm::GemmShape;

  return {
    CutlassConvInstance<half, Sm60, GemmShape<256, 32, 8>, GemmShape<64, 32, 8>, 2>::get(),
    CutlassConvInstance<half, Sm60, GemmShape<256, 64, 8>, GemmShape<64, 64, 8>, 2>::get(),
  };
}

} // namespace oidn