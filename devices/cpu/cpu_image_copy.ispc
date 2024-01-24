// Copyright 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_accessor.isph"

struct CPUImageCopyKernel
{
  uniform ImageAccessor src;
  uniform ImageAccessor dst;
};

export void CPUImageCopyKernel_run(const uniform CPUImageCopyKernel* uniform self, uniform int h)
{
  foreach (w = 0 ... self->dst.W)
  {
    vec3f value = Image_get3(self->src, h, w);
    Image_set3(self->dst, h, w, value);
  }
}