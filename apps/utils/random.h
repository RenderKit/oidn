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
    OIDN_INLINE Random(uint32_t seed = 1) : state(seed) {}

    OIDN_INLINE void reset(uint32_t seed = 1)
    {
      state = (seed * 8191) ^ 140167;
    }

    OIDN_INLINE void next()
    {
      const uint32_t multiplier = 1664525;
      const uint32_t increment  = 1013904223;
      state = multiplier * state + increment;
    }

    OIDN_INLINE uint32_t get1ui()
    {
      next();
      return state;
    }

    OIDN_INLINE int get1i()
    {
      next();
      return state;
    }

    OIDN_INLINE float get1f()
    {
      next();
      return to_float_unorm(state);
    }
  };

} // namespace oidn