// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_upsample.isph"

#if TARGET_WIDTH * 2 != BLOCKC_BYTESIZE
  #error "Unsupported ISPC target width"
#endif

export void CPUUpsampleKernel_run_f16(const uniform CPUUpsampleKernel* uniform self,
                                      uniform int cb, uniform int h)
{
  CPUUpsampleKernel_run<float16>(self, cb, h);
}