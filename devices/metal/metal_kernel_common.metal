// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "metal_kernel_constants.h"
#include "metal_kernel_common.h"

float nan_to_zero(float value)
{
  return value == NAN ? 0 : value;
}
