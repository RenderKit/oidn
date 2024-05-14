// Copyright 2009 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../include/OpenImageDenoise/config.h"

// -------------------------------------------------------------------------------------------------
// Macros
// -------------------------------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64)
  #define OIDN_ARCH_X64
#elif defined(__aarch64__) || defined(_M_ARM64)
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

#if defined(__METAL_VERSION__)
  #define OIDN_COMPILE_METAL_DEVICE
#endif
#if defined(OIDN_COMPILE_METAL_HOST) || defined(OIDN_COMPILE_METAL_DEVICE)
  #define OIDN_COMPILE_METAL
#endif

#if defined(OIDN_COMPILE_SYCL_DEVICE) || defined(OIDN_COMPILE_CUDA_DEVICE) || \
    defined(OIDN_COMPILE_HIP_DEVICE)  || defined(OIDN_COMPILE_METAL_DEVICE)
  #define OIDN_COMPILE_DEVICE
#endif

#if defined(_WIN32)
  // Windows
  #define oidn_inline __forceinline
  #define oidn_noinline __declspec(noinline)
#else
  // Unix
  #define oidn_inline inline __attribute__((always_inline))
  #define oidn_noinline __attribute__((noinline))
#endif

#ifndef UNUSED
  #define UNUSED(x) ((void)x)
#endif
#ifndef MAYBE_UNUSED
  #define MAYBE_UNUSED(x) UNUSED(x)
#endif

#if defined(OIDN_COMPILE_CUDA) || defined(OIDN_COMPILE_HIP)
  #define oidn_device __device__
  #define oidn_device_inline __device__ oidn_inline
  #define oidn_host_device __host__ __device__
  #define oidn_host_device_inline __host__ __device__ oidn_inline
  #define oidn_constant
  #define oidn_global
  #define oidn_local
  #define oidn_private
#else
  #define oidn_device
  #define oidn_device_inline oidn_inline
  #define oidn_host_device
  #define oidn_host_device_inline oidn_inline
  #if defined(OIDN_COMPILE_METAL_DEVICE)
    #define oidn_constant constant
    #define oidn_global device
    #define oidn_local threadgroup
    #define oidn_private thread
  #else
    #define oidn_constant
    #define oidn_global
    #define oidn_local
    #define oidn_private
  #endif
#endif

// Helper string macros
#define _OIDN_TO_STRING(a) #a
#define OIDN_TO_STRING(a) _OIDN_TO_STRING(a)

#define _OIDN_CONCAT(a, b) a##b
#define OIDN_CONCAT(a, b) _OIDN_CONCAT(a, b)

// -------------------------------------------------------------------------------------------------
// Includes
// -------------------------------------------------------------------------------------------------

#if defined(OIDN_COMPILE_METAL_DEVICE)
  #include <metal_stdlib>
#else
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
    #include <TargetConditionals.h>
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
  #include <stdexcept>

  #if defined(OIDN_COMPILE_SYCL)
    #include <sycl/sycl.hpp>
    #include <sycl/ext/intel/esimd.hpp>
  #elif defined(OIDN_COMPILE_CUDA)
    #include <cuda_fp16.h>
  #elif defined(OIDN_COMPILE_HIP)
    #include <hip/hip_runtime.h>
    #include <hip/hip_fp16.h>
  #else
    #include "half.h"
  #endif
#endif

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_COMPILE_SYCL)
  namespace syclx  = sycl::ext::intel::experimental;
  namespace esimd  = sycl::ext::intel::esimd;
  namespace esimdx = sycl::ext::intel::experimental::esimd;

  using sycl::half;
#endif

#if !defined(OIDN_COMPILE_METAL_DEVICE)
  template<bool B, class T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;
#endif

  // -----------------------------------------------------------------------------------------------
  // Common functions
  // -----------------------------------------------------------------------------------------------

  template<typename T>
  oidn_host_device_inline constexpr T min(T a, T b) { return (b < a) ? b : a; }

  template<typename T>
  oidn_host_device_inline constexpr T max(T a, T b) { return (a < b) ? b : a; }

  template<typename T>
  oidn_host_device_inline constexpr T clamp(T x, T minVal, T maxVal)
  {
    return min(max(x, minVal), maxVal);
  }

  // Returns ceil(a / b) for non-negative integers
  template<typename Int, typename IntB>
  oidn_host_device_inline constexpr Int ceil_div(Int a, IntB b)
  {
    //assert(a >= 0);
    //assert(b > 0);
    return (a + b - 1) / b;
  }

  // Returns a rounded up to multiple of b
  template<typename Int, typename IntB>
  oidn_host_device_inline constexpr Int round_up(Int a, IntB b)
  {
    return ceil_div(a, b) * b;
  }

  // Returns the smallest integer larger than or equal to a which has remainder c when divided by b
  template<typename Int, typename IntB>
  oidn_host_device_inline constexpr Int round_up(Int a, IntB b, IntB c)
  {
    //assert(a >= 0);
    //assert(b > 0);
    //assert(c >= 0 && c < b);
    return (a + b - c - 1) / b * b + c;
  }

  // Returns the greatest common divisor of a and b
  template<typename Int>
  oidn_host_device_inline Int gcd(Int a, Int b)
  {
    while (b != 0)
    {
      const Int t = b;
      b = a % b;
      a = t;
    }
    return a;
  }

  // Returns the least common multiple of a and b
  template<typename Int>
  oidn_host_device_inline Int lcm(Int a, Int b)
  {
    return (a * b) / gcd(a, b);
  }

  // -----------------------------------------------------------------------------------------------
  // Data type
  // -----------------------------------------------------------------------------------------------

  // Data types sorted by precision in ascending order
  enum class DataType
  {
    Void,
    UInt8,
    Float16,
    Float32,
  };

#if !defined(OIDN_COMPILE_METAL_DEVICE)

  std::ostream& operator <<(std::ostream& sm, DataType dataType);

  // -----------------------------------------------------------------------------------------------
  // Memory allocation
  // -----------------------------------------------------------------------------------------------

  struct SizeAndAlignment
  {
    size_t size;
    size_t alignment;
  };

  constexpr size_t memoryAlignment = 256;

  void* alignedMalloc(size_t size, size_t alignment = memoryAlignment);
  void alignedFree(void* ptr);

  // -----------------------------------------------------------------------------------------------
  // String functions
  // -----------------------------------------------------------------------------------------------

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

  inline std::string toLower(const std::string& str)
  {
    std::string result = str;
    std::transform(str.begin(), str.end(), result.begin(), ::tolower);
    return result;
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

  inline bool isEnvVar(const std::string& name)
  {
    auto* str = getenv(name.c_str());
    return (str != nullptr);
  }

  template<typename T>
  inline bool getEnvVar(const std::string& name, T& value)
  {
    auto* str = getenv(name.c_str());
    bool found = (str != nullptr);
    if (found)
      value = fromString<T>(str);
    return found;
  }

  template<typename T>
  inline T getEnvVarOrDefault(const std::string& name, const T& defaultValue)
  {
    T value = defaultValue;
    getEnvVar(name, value);
    return value;
  }

  template<typename T>
  inline bool setEnvVar(const std::string& name, const T& value, bool overwrite)
  {
    const std::string valueStr = toString(value);
  #if defined(_WIN32)
    if (overwrite || !isEnvVar(name))
      return _putenv_s(name.c_str(), valueStr.c_str()) == 0;
    else
      return true;
  #else
    return setenv(name.c_str(), valueStr.c_str(), overwrite) == 0;
  #endif
  }

  // -----------------------------------------------------------------------------------------------
  // System information
  // -----------------------------------------------------------------------------------------------

  std::string getOSName();
  std::string getCompilerName();
  std::string getBuildName();

#endif // !defined(OIDN_COMPILE_METAL_DEVICE)

OIDN_NAMESPACE_END

