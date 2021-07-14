// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "image.h"
#include "output_copy_ispc.h"

namespace oidn {

  // Output copy function
  void outputCopy(const Ref<Device>& device,
                  const Image& src,
                  const Image& dst);

} // namespace oidn
