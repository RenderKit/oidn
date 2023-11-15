// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

OIDN_NAMESPACE_BEGIN
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
  template<typename T> oidn_host_device_inline T min(T a, T b) { return ::min(a, b); }
  template<typename T> oidn_host_device_inline T max(T a, T b) { return ::max(a, b); }
  using ::isfinite;
  using ::isnan;
  using ::pow;
  using ::log;
  using ::log2;
  using ::exp;
  using ::exp2;
#elif defined(OIDN_COMPILE_METAL_DEVICE)
  // Use the Metal math functions
  using metal::min;
  using metal::max;
  using metal::isfinite;
  using metal::isnan;
  using metal::pow;
  using metal::log;
  using metal::log2;
  using metal::exp;
  using metal::exp2;
#else
  using OIDN_NAMESPACE::min;
  using OIDN_NAMESPACE::max;
  using std::isfinite;
  using std::isnan;
  using std::pow;
  using std::log;
  using std::log2;
  using std::exp;
  using std::exp2;
#endif

  // CUDA and HIP do not provide min/max overloads for half
#if defined(OIDN_COMPILE_CUDA_DEVICE) && (__CUDA_ARCH__ >= 800)
  oidn_device_inline half min(half a, half b) { return __hmin(a, b); }
  oidn_device_inline half max(half a, half b) { return __hmax(a, b); }
#elif (defined(OIDN_COMPILE_CUDA_DEVICE) && (__CUDA_ARCH__ >= 530)) || defined(OIDN_COMPILE_HIP_DEVICE)
  oidn_device_inline half min(half a, half b) { return (b < a) ? b : a; }
  oidn_device_inline half max(half a, half b) { return (a < b) ? b : a; }
#endif

  template<typename T>
  oidn_host_device_inline T clamp(T x, T minVal, T maxVal)
  {
    return min(max(x, minVal), maxVal);
  }

  oidn_host_device_inline float to_float_unorm(uint32_t x)
  {
    return float(x) * 2.3283064365386962890625e-10f; // x / 2^32
  }

  // Maps nan to zero
  oidn_host_device_inline float nan_to_zero(float x)
  {
    return isnan(x) ? 0.f : x;
  }

} // namespace math
OIDN_NAMESPACE_END
