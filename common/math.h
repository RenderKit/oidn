// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "platform.h"

namespace oidn {
namespace math {

#if defined(OIDN_COMPILE_SYCL_DEVICE)
  // Use the SYCL math functions
  using sycl::min;
  using sycl::max;
  using sycl::isfinite;
  using sycl::isnan;
  using sycl::pow;
  using sycl::log;
  using sycl::log2;
  using sycl::exp;
  using sycl::exp2;
#elif defined(OIDN_COMPILE_CUDA_DEVICE) || defined(OIDN_COMPILE_HIP_DEVICE)
  // Use the CUDA/HIP math functions
  using ::min;
  using ::max;
  using ::isfinite;
  using ::isnan;
  using ::pow;
  using ::log;
  using ::log2;
  using ::exp;
  using ::exp2;
#else
  using oidn::min;
  using oidn::max;
  using std::isfinite;
  using std::isnan;
  using std::pow;
  using std::log;
  using std::log2;
  using std::exp;
  using std::exp2;
#endif

#if defined(OIDN_COMPILE_CUDA_DEVICE)
  // CUDA currently does not provide min/max overloads for half float
#if __CUDA_ARCH__ >= 800
  OIDN_DEVICE_INLINE half min(half a, half b) { return __hmin(a, b); }
  OIDN_DEVICE_INLINE half max(half a, half b) { return __hmax(a, b); }
#elif __CUDA_ARCH__ >= 530
  OIDN_DEVICE_INLINE half min(half a, half b) { return (b < a) ? b : a; }
  OIDN_DEVICE_INLINE half max(half a, half b) { return (a < b) ? b : a; }
#endif
#endif

  template<typename T>
  OIDN_HOST_DEVICE_INLINE T clamp(T x, T minVal, T maxVal)
  {
    return min(max(x, minVal), maxVal);
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

} // namespace math
} // namespace oidn
