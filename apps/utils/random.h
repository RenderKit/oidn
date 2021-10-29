// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/math.h"

namespace oidn {

  // Simple and very fast LCG random number generator
  class Random
  {
  private:
    uint32_t state;

  public:
    __forceinline Random(uint32_t seed = 1) : state(seed) {}

    __forceinline void reset(uint32_t seed = 1)
    {
      state = (seed * 8191) ^ 140167;
    }

    __forceinline void next()
    {
      const uint32_t multiplier = 1664525;
      const uint32_t increment  = 1013904223;
      state = multiplier * state + increment;
    }

    __forceinline uint32_t get1ui()
    {
      next();
      return state;
    }

    __forceinline int get1i()
    {
      next();
      return state;
    }

    __forceinline float get1f()
    {
      next();
      return to_float_unorm(state);
    }
  };

} // namespace oidn