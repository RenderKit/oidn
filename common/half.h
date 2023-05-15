// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "include/OpenImageDenoise/config.h"
#include <cstdint>

OIDN_NAMESPACE_BEGIN

  float half_to_float(int16_t x);
  int16_t float_to_half(float x);

  // Minimal half data type
  class half
  {
  public:
    half() = default;
    half(const half& h) : x(h.x) {}
    half(float f) : x(float_to_half(f)) {}

    half& operator =(const half& h) { x = h.x; return *this; }
    half& operator =(float f) { x = float_to_half(f); return *this; }

    operator float() const { return half_to_float(x); }

  private:
    int16_t x;
  };

OIDN_NAMESPACE_END