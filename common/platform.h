// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

// ---------------------------------------------------------------------------
// Macros
// ---------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64)
  #define OIDN_X64
#elif defined(__aarch64__)
  #define OIDN_ARM64
#endif

#if defined(SYCL_LANGUAGE_VERSION)
  #define OIDN_SYCL
#endif

#if defined(_WIN32)
  // Windows
  #define OIDN_INLINE OIDN_INLINE
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

#define OIDN_DEVICE
#define OIDN_DEVICE_INLINE OIDN_INLINE
#define OIDN_HOST_DEVICE
#define OIDN_HOST_DEVICE_INLINE OIDN_INLINE

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
  #include <windows.h>
#elif defined(__APPLE__)
  #include <sys/sysctl.h>
#endif

#if defined(OIDN_X64)
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
#include <cmath>
#include <cfloat>
#include <string>
#include <sstream>
#include <iostream>
#include <cassert>

#if defined(OIDN_SYCL)
  #include <CL/sycl.hpp>
  #include <sycl/ext/intel/esimd.hpp>
#endif

#include "include/OpenImageDenoise/oidn.hpp"

namespace oidn {

  // Introduce all names from the API namespace
  OIDN_NAMESPACE_USING

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

  using std::min;
  using std::max;

  template<typename T>
  OIDN_DEVICE_INLINE T clamp(const T& x, const T& minVal, const T& maxVal)
  {
    return min(max(x, minVal), maxVal);
  }

  constexpr size_t memoryAlignment = 128;

  void* alignedMalloc(size_t size, size_t alignment = memoryAlignment);
  void alignedFree(void* ptr);

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
  // FP16
  // ---------------------------------------------------------------------------

  float half_to_float(int16_t x);
  int16_t float_to_half(float x);

  #if defined(OIDN_SYCL)
    using sycl::half;
  #else
    // Minimal half data type
    class half
    {
    private:
      int16_t x;

    public:
      half() = default;
      half(const half& h) : x(h.x) {}
      half(float f) : x(float_to_half(f)) {}

      half& operator =(const half& h) { x = h.x; return *this; }
      half& operator =(float f) { x = float_to_half(f); return *this; }
      
      operator float() const { return half_to_float(x); }
    };
  #endif

  // ---------------------------------------------------------------------------
  // System information
  // ---------------------------------------------------------------------------

#if defined(OIDN_X64)
  enum class ISA
  {
    SSE41,
    AVX2,
    AVX512_CORE
  };

  bool isISASupported(ISA isa);
#endif

  std::string getPlatformName();
  std::string getCompilerName();
  std::string getBuildName();

} // namespace oidn

