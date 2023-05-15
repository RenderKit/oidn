// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_accessor.isph"

struct CPUImageCopyKernel
{
  uniform ImageAccessor src;
  uniform ImageAccessor dst;
};

export void CPUImageCopyKernel_run(uniform CPUImageCopyKernel* uniform self, uniform int h)
{
  foreach (w = 0 ... self->dst.W)
  {
    vec3f value = get3f(self->src, h, w);
    set3f(self->dst, h, w, value);
  }
}