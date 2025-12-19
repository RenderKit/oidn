// Copyright 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_pool.isph"

#if TARGET_WIDTH * 4 != BLOCKC_BYTESIZE
  #error "Unsupported ISPC target width"
#endif

export void CPUPoolKernel_run_f32(const uniform CPUPoolKernel* uniform self,
                                  uniform int cb, uniform int h)
{
  CPUPoolKernel_run<float>(self, cb, h);
}