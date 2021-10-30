// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "platform.h"

#if defined(OIDN_X64)
  #include "mkl-dnn/src/cpu/x64/xbyak/xbyak_util.h"
#endif

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
  // FP16
  // ---------------------------------------------------------------------------

  // https://gist.github.com/rygorous/2156668
  namespace
  {
    typedef unsigned int uint;

    union FP32
    {
      uint u;
      float f;
      struct
      {
        uint Mantissa : 23;
        uint Exponent : 8;
        uint Sign : 1;
      };
    };

    union FP16
    {
      unsigned short u;
      struct
      {
        uint Mantissa : 10;
        uint Exponent : 5;
        uint Sign : 1;
      };
    };

    // Original ISPC reference version; this always rounds ties up.
    FP16 float_to_half(FP32 f)
    {
      FP16 o = { 0 };

      // Based on ISPC reference code (with minor modifications)
      if (f.Exponent == 0) // Signed zero/denormal (which will underflow)
        o.Exponent = 0;
      else if (f.Exponent == 255) // Inf or NaN (all exponent bits set)
      {
        o.Exponent = 31;
        o.Mantissa = f.Mantissa ? 0x200 : 0; // NaN->qNaN and Inf->Inf
      }
      else // Normalized number
      {
        // Exponent unbias the single, then bias the halfp
        int newexp = f.Exponent - 127 + 15;
        if (newexp >= 31) // Overflow, return signed infinity
          o.Exponent = 31;
        else if (newexp <= 0) // Underflow
        {
          if ((14 - newexp) <= 24) // Mantissa might be non-zero
          {
            uint mant = f.Mantissa | 0x800000; // Hidden 1 bit
            o.Mantissa = mant >> (14 - newexp);
            if ((mant >> (13 - newexp)) & 1) // Check for rounding
              o.u++; // Round, might overflow into exp bit, but this is OK
          }
        }
        else
        {
          o.Exponent = newexp;
          o.Mantissa = f.Mantissa >> 13;
          if (f.Mantissa & 0x1000) // Check for rounding
            o.u++; // Round, might overflow to inf, this is OK
        }
      }

      o.Sign = f.Sign;
      return o;
    }

    FP32 half_to_float(FP16 h)
    {
      static const FP32 magic = { 113 << 23 };
      static const uint shifted_exp = 0x7c00 << 13; // exponent mask after shift
      FP32 o;

      o.u = (h.u & 0x7fff) << 13;     // exponent/mantissa bits
      uint exp = shifted_exp & o.u;   // just the exponent
      o.u += (127 - 15) << 23;        // exponent adjust

      // handle exponent special cases
      if (exp == shifted_exp) // Inf/NaN?
        o.u += (128 - 16) << 23;    // extra exp adjust
      else if (exp == 0) // Zero/Denormal?
      {
        o.u += 1 << 23;             // extra exp adjust
        o.f -= magic.f;             // renormalize
      }

      o.u |= (h.u & 0x8000) << 16;    // sign bit
      return o;
    }
  }

  float half_to_float(int16_t x)
  {
    FP16 fp16;
    fp16.u = (unsigned short)x;
    return half_to_float(fp16).f;
  }

  int16_t float_to_half(float x)
  {
    FP32 fp32;
    fp32.f = x;
    return (int16_t)float_to_half(fp32).u;
  }

  // ---------------------------------------------------------------------------
  // System information
  // ---------------------------------------------------------------------------

#if defined(OIDN_X64)
  bool isISASupported(ISA isa)
  {
    using Xbyak::util::Cpu;
    static Cpu cpu;
    
    switch (isa)
    {
    case ISA::SSE41:
      return cpu.has(Cpu::tSSE41);
    case ISA::AVX2:
      return cpu.has(Cpu::tAVX2);
    case ISA::AVX512_CORE:
      return cpu.has(Cpu::tAVX512F)  && cpu.has(Cpu::tAVX512BW) &&
             cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ);
    default:
      return false;
    }
  }
#endif

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
