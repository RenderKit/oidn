// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"

namespace oidn {

#if defined(OIDN_CUDA) || defined(OIDN_HIP)
  // Make sure to use the CUDA/HIP math functions
  using ::isfinite;
  using ::isnan;
  using ::pow;
  using ::log;
  using ::log2;
  using ::exp;
  using ::exp2;
#else
  using std::isfinite;
  using std::isnan;
  using std::pow;
  using std::log;
  using std::log2;
  using std::exp;
  using std::exp2;
#endif

  // Returns ceil(a / b) for non-negative integers
  template<typename Int, typename IntB>
  OIDN_HOST_DEVICE_INLINE constexpr Int ceil_div(Int a, IntB b)
  {
    //assert(a >= 0);
    //assert(b > 0);
    return (a + b - 1) / b;
  }

  // Returns a rounded up to multiple of b
  template<typename Int, typename IntB>
  OIDN_HOST_DEVICE_INLINE constexpr Int round_up(Int a, IntB b)
  {
    return ceil_div(a, b) * b;
  }

  OIDN_HOST_DEVICE_INLINE float to_float_unorm(uint32_t x)
  {
    return float(x) * 2.3283064365386962890625e-10f; // x / 2^32
  }

  // Maps nan to zero
  OIDN_HOST_DEVICE_INLINE float nan_to_zero(float x)
  {
    return isnan(x) ? 0.f : x;
  }

} // namespace oidn
