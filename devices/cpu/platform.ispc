// Copyright 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "platform.isph"

// -----------------------------------------------------------------------------------------------
// System information
// -----------------------------------------------------------------------------------------------

enum CPUArch
{
  CPUArch_Unknown,
  CPUArch_SSE2,
  CPUArch_SSE4,
  CPUArch_AVX2,
  CPUArch_AVX512,
  CPUArch_NEON
};

export uniform CPUArch getCPUArch()
{
#if defined(ISPC_TARGET_SSE2)
  return CPUArch_SSE2;
#elif defined(ISPC_TARGET_SSE4)
  return CPUArch_SSE4;
#elif defined(ISPC_TARGET_AVX2)
  return CPUArch_AVX2;
#elif defined(ISPC_TARGET_AVX512SKX) || defined(ISPC_TARGET_AVX512SPR)
  return CPUArch_AVX512;
#elif defined(ISPC_TARGET_NEON)
  return CPUArch_NEON;
#endif
  return CPUArch_Unknown;
}