// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/math.h"

namespace oidn {

  template<typename T>
  struct vec3
  {
    T x, y, z;

    __forceinline vec3() {}
    __forceinline vec3(T x, T y, T z) : x(x), y(y), z(z) {}
  };

  using vec3f = vec3<float>;

} // namespace oidn