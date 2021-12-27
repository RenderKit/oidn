// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "image_accessor.h"

namespace oidn {

  template<typename T>
  struct ImageCopy
  {
    // Source
    ImageAccessor<T> src;

    // Destination
    ImageAccessor<T> dst;

    OIDN_DEVICE_INLINE void operator ()(int h, int w) const
    {
      // Load
      vec3<T> value = src.get3(h, w);

      // Store
      dst.set3(h, w, value);
    }
  };

} // namespace oidn
