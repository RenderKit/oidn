// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"

namespace oidn {

  using std::isfinite;
  using std::isnan;
  using std::pow;
  using std::log;
  using std::exp;

  // Returns ceil(a / b) for non-negative integers
  template<typename Int, typename IntB>
  __forceinline constexpr Int ceil_div(Int a, IntB b)
  {
    //assert(a >= 0);
    //assert(b > 0);
    return (a + b - 1) / b;
  }

  // Returns a rounded up to multiple of b
  template<typename Int, typename IntB>
  __forceinline constexpr Int round_up(Int a, IntB b)
  {
    return ceil_div(a, b) * b;
  }

  __forceinline float to_float_unorm(uint32_t x)
  {
    return float(x) * 2.3283064365386962890625e-10f; // x / 2^32
  }

  // Maps nan to zero
  __forceinline float nan_to_zero(float x)
  {
    return isnan(x) ? 0.f : x;
  }

} // namespace oidn
