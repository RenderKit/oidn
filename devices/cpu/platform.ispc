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
  CPUArch_AVX512_AMXFP16,
  CPUArch_NEON
};

export uniform CPUArch getCPUArch()
{
#if defined(OIDN_ARCH_AVX512_AMXFP16)
  return CPUArch_AVX512_AMXFP16;
#elif defined(OIDN_ARCH_AVX512)
  return CPUArch_AVX512;
#elif defined(OIDN_ARCH_AVX2)
  return CPUArch_AVX2;
#elif defined(OIDN_ARCH_SSE4)
  return CPUArch_SSE4;
#elif defined(OIDN_ARCH_SSE2)
  return CPUArch_SSE2;
#elif defined(OIDN_ARCH_NEON)
  return CPUArch_NEON;
#else
  #error "Unsupported architecture"
#endif
}