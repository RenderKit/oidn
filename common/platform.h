// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64)
  #define OIDN_ARCH_X64
#elif defined(__aarch64__)
  #define OIDN_ARCH_ARM64
#endif

#if defined(SYCL_LANGUAGE_VERSION)
  #define OIDN_COMPILE_SYCL
#endif
#if defined(__SYCL_DEVICE_ONLY__)
  #define OIDN_COMPILE_SYCL_DEVICE
#endif

#if defined(__CUDACC__)
  #define OIDN_COMPILE_CUDA
#endif
#if defined(__CUDA_ARCH__)
  #define OIDN_COMPILE_CUDA_DEVICE
#endif

#if defined(__HIPCC__)
  #define OIDN_COMPILE_HIP
#endif
#if defined(__HIP_DEVICE_COMPILE__)
  #define OIDN_COMPILE_HIP_DEVICE
#endif

#if defined(OIDN_COMPILE_SYCL_DEVICE) || defined(OIDN_COMPILE_CUDA_DEVICE) || defined(OIDN_COMPILE_HIP_DEVICE)
  #define OIDN_COMPILE_DEVICE
#endif

#if defined(_WIN32)
  // Windows
  #define OIDN_INLINE __forceinline
  #define OIDN_NOINLINE __declspec(noinline)
#else
  // Unix
  #define OIDN_INLINE inline __attribute__((always_inline))
  #define OIDN_NOINLINE __attribute__((noinline))
#endif

#ifndef UNUSED
  #define UNUSED(x) ((void)x)
#endif
#ifndef MAYBE_UNUSED
  #define MAYBE_UNUSED(x) UNUSED(x)
#endif

#if defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)
  #define OIDN_DEVICE __device__
  #define OIDN_DEVICE_INLINE __device__ OIDN_INLINE
  #define OIDN_HOST_DEVICE __host__ __device__
  #define OIDN_HOST_DEVICE_INLINE __host__ __device__ OIDN_INLINE
  #define OIDN_SHARED __shared__
#else
  #define OIDN_DEVICE
  #define OIDN_DEVICE_INLINE OIDN_INLINE
  #define OIDN_HOST_DEVICE
  #define OIDN_HOST_DEVICE_INLINE OIDN_INLINE
  #define OIDN_SHARED
#endif

#define OIDN_CONCAT2(a, b) a##b
#define OIDN_CONCAT(a, b) OIDN_CONCAT2(a, b)

// ---------------------------------------------------------------------------
// Includes
// ---------------------------------------------------------------------------

#if defined(_WIN32)
  #if !defined(WIN32_LEAN_AND_MEAN)
    #define WIN32_LEAN_AND_MEAN
  #endif
  #if !defined(NOMINMAX)
    #define NOMINMAX
  #endif
  #include <Windows.h>
#elif defined(__APPLE__)
  #include <sys/sysctl.h>
#endif

#if defined(OIDN_ARCH_X64)
  #include <xmmintrin.h>
  #include <pmmintrin.h>
#endif

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <climits>
#include <cstring>
#include <limits>
#include <atomic>
#include <algorithm>
#include <memory>
#include <array>
#include <type_traits>
#include <cmath>
#include <cfloat>
#include <string>
#include <sstream>
#include <iostream>
#include <cassert>

#if defined(OIDN_COMPILE_SYCL)
  #include <sycl/sycl.hpp>
  #include <sycl/ext/intel/esimd.hpp>
#endif

#if defined(OIDN_COMPILE_CUDA)
  #include <cuda_fp16.h>
#elif defined(OIDN_COMPILE_HIP)
  #include <hip/hip_runtime.h>
  #include <hip/hip_fp16.h>
#elif !defined(OIDN_COMPILE_SYCL)
  #include "half.h"
#endif

namespace oidn {

#if defined(OIDN_COMPILE_SYCL)
  namespace syclx  = sycl::ext::intel::experimental;
  namespace esimd  = sycl::ext::intel::esimd;
  namespace esimdx = sycl::ext::intel::experimental::esimd;

  using sycl::half;
#endif

  template<bool B, class T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;

  // ---------------------------------------------------------------------------
  // Error handling and debugging
  // ---------------------------------------------------------------------------

  struct Verbose
  {
    int verbose;

    Verbose(int v = 0) : verbose(v) {}
    OIDN_INLINE bool isVerbose(int v = 1) const { return v <= verbose; }
  };

  #define OIDN_WARNING(message) { if (isVerbose()) std::cerr << "Warning: " << message << std::endl; }
  #define OIDN_FATAL(message) throw std::runtime_error(message);

  // ---------------------------------------------------------------------------
  // Common functions
  // ---------------------------------------------------------------------------

  template<typename T>
  OIDN_HOST_DEVICE_INLINE constexpr T min(T a, T b) { return (b < a) ? b : a; }

  template<typename T>
  OIDN_HOST_DEVICE_INLINE constexpr T max(T a, T b) { return (a < b) ? b : a; }

  template<typename T>
  OIDN_HOST_DEVICE_INLINE constexpr T clamp(T x, T minVal, T maxVal)
  {
    return min(max(x, minVal), maxVal);
  }

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

  // ---------------------------------------------------------------------------
  // Memory allocation
  // ---------------------------------------------------------------------------

  constexpr size_t memoryAlignment = 128;

  void* alignedMalloc(size_t size, size_t alignment = memoryAlignment);
  void alignedFree(void* ptr);

  // ---------------------------------------------------------------------------
  // String functions
  // ---------------------------------------------------------------------------

  template<typename T>
  inline std::string toString(const T& a)
  {
    std::stringstream sm;
    sm << a;
    return sm.str();
  }

  template<typename T>
  inline T fromString(const std::string& str)
  {
    std::stringstream sm(str);
    T a{};
    sm >> a;
    return a;
  }

  template<>
  inline std::string fromString(const std::string& str)
  {
    return str;
  }

#if defined(__APPLE__)
  template<typename T>
  inline bool getSysctl(const char* name, T& value)
  {
    int64_t result = 0;
    size_t size = sizeof(result);

    if (sysctlbyname(name, &result, &size, nullptr, 0) != 0)
      return false;

    value = T(result);
    return true;
  }
#endif

  template<typename T>
  inline bool getEnvVar(const std::string& name, T& value)
  {
    auto* str = getenv(name.c_str());
    bool found = (str != nullptr);
    if (found)
      value = fromString<T>(str);
    return found;
  }

  inline bool isEnvVar(const std::string& name)
  {
    auto* str = getenv(name.c_str());
    return (str != nullptr);
  }

  // ---------------------------------------------------------------------------
  // System information
  // ---------------------------------------------------------------------------

  std::string getOSName();
  std::string getCompilerName();
  std::string getBuildName();

} // namespace oidn

