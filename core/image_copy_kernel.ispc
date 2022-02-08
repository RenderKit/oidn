// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "image_accessor.isph"

struct ImageCopyKernel
{
  uniform ImageAccessor src;
  uniform ImageAccessor dst;
};

export void ImageCopyKernel_run(uniform ImageCopyKernel* uniform self, uniform int h)
{
  foreach (w = 0 ... self->dst.W)
  {
    vec3f value = get3f(self->src, h, w);
    set3f(self->dst, h, w, value);
  }
}