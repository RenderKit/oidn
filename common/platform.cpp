// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "platform.h"

namespace oidn {

  // ---------------------------------------------------------------------------
  // Common functions
  // ---------------------------------------------------------------------------

  void* alignedMalloc(size_t size, size_t alignment)
  {
    if (size == 0)
      return nullptr;

    assert((alignment & (alignment-1)) == 0);
  #if defined(OIDN_X64)
    void* ptr = _mm_malloc(size, alignment);
  #else
    void* ptr;
    if (posix_memalign(&ptr, max(alignment, sizeof(void*)), size) != 0)
      ptr = nullptr;
  #endif

    if (ptr == nullptr)
      throw std::bad_alloc();

    return ptr;
  }

  void alignedFree(void* ptr)
  {
    if (ptr)
    #if defined(OIDN_X64)
      _mm_free(ptr);
    #else
      free(ptr);
    #endif
  }

  // ---------------------------------------------------------------------------
  // System information
  // ---------------------------------------------------------------------------

  std::string getPlatformName()
  {
    std::string name;

  #if defined(__linux__)
    name = "Linux";
  #elif defined(__FreeBSD__)
    name = "FreeBSD";
  #elif defined(__CYGWIN__)
    name = "Cygwin";
  #elif defined(_WIN32)
    name = "Windows";
  #elif defined(__APPLE__)
    name = "macOS";
  #elif defined(__unix__)
    name = "Unix";
  #else
    return "Unknown";
  #endif

  #if defined(__x86_64__) || defined(_M_X64) || defined(__ia64__) || defined(__aarch64__)
    name += " (64-bit)";
  #else
    name += " (32-bit)";
  #endif

    return name;
  }

  std::string getCompilerName()
  {
  #if defined(__INTEL_COMPILER)
    int major = __INTEL_COMPILER / 100 % 100;
    int minor = __INTEL_COMPILER % 100 / 10;
    std::string version = "Intel Compiler ";
    version += toString(major);
    version += "." + toString(minor);
  #if defined(__INTEL_COMPILER_UPDATE)
    version += "." + toString(__INTEL_COMPILER_UPDATE);
  #endif
    return version;
  #elif defined(__clang__)
    return "Clang " __clang_version__;
  #elif defined(__GNUC__)
    return "GCC " __VERSION__;
  #elif defined(_MSC_VER)
    std::string version = toString(_MSC_FULL_VER);
    version.insert(4, ".");
    version.insert(9, ".");
    version.insert(2, ".");
    return "Visual C++ Compiler " + version;
  #else
    return "Unknown";
  #endif
  }

  std::string getBuildName()
  {
  #if defined(NDEBUG)
    return "Release";
  #else
    return "Debug";
  #endif
  }

} // namespace oidn
