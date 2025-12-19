// Copyright 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_input_process.isph"

export void CPUInputProcessKernel_run_f32(const uniform CPUInputProcessKernel* uniform self,
                                          uniform int hDst)
{
  CPUInputProcessKernel_run<float>(self, hDst);
}