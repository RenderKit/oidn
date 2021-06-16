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

#if defined(_WIN32)
  // Windows
  #if !defined(__noinline)
    #define __noinline __declspec(noinline)
  #endif
#else
  // Unix
  #if !defined(__forceinline)
    #define __forceinline inline __attribute__((always_inline))
  #endif
  #if !defined(__noinline)
    #define __noinline __attribute__((noinline))
  #endif
#endif

#ifndef UNUSED
  #define UNUSED(x) ((void)x)
#endif
#ifndef MAYBE_UNUSED
  #define MAYBE_UNUSED(x) UNUSED(x)
#endif

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
#include <string>
#include <sstream>
#include <iostream>
#include <cassert>
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
    __forceinline bool isVerbose(int v = 1) const { return v <= verbose; }
  };

  #define OIDN_WARNING(message) { if (isVerbose()) std::cerr << "Warning: " << message << std::endl; }
  #define OIDN_FATAL(message) throw std::runtime_error(message);

  // ---------------------------------------------------------------------------
  // Common functions
  // ---------------------------------------------------------------------------

  using std::min;
  using std::max;

  template<typename T>
  __forceinline T clamp(const T& value, const T& minValue, const T& maxValue)
  {
    return min(max(value, minValue), maxValue);
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
  // System information
  // ---------------------------------------------------------------------------

  std::string getPlatformName();
  std::string getCompilerName();
  std::string getBuildName();

} // namespace oidn

