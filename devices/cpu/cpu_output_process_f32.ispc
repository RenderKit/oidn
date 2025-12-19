// Copyright 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_output_process.isph"

export void CPUOutputProcessKernel_run_f32(const uniform CPUOutputProcessKernel* uniform self,
                                           uniform int h)
{
  CPUOutputProcessKernel_run<float>(self, h);
}