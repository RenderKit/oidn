// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "half.h"

OIDN_NAMESPACE_BEGIN

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

OIDN_NAMESPACE_END