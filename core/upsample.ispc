// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"

struct Upsample
{
  uniform TensorAccessor3D src;
  uniform TensorAccessor3D dst;
};

export void Upsample_kernel(uniform Upsample* uniform self, uniform int ck, uniform int h)
{
  const uniform size_t H = (size_t)self->src.H;
  const uniform size_t W = (size_t)self->src.W;

  const uniform size_t offset = (ck*H + h) * (W*B);
  uniform float* const uniform srcPtr_line  = self->src.ptr + offset;
  uniform float* const uniform dstPtr_line0 = self->dst.ptr + offset * 4;
  uniform float* const uniform dstPtr_line1 = dstPtr_line0 + W*2*B; // next line

  for (uniform size_t w = 0; w < W; ++w)
  {
    // Load vector
    const float value = *((varying float* uniform)&srcPtr_line[w*B]);

    // Store vector 2x2
    streaming_store(&dstPtr_line0[w*2*B  ], value);
    streaming_store(&dstPtr_line0[w*2*B+B], value);
    streaming_store(&dstPtr_line1[w*2*B  ], value);
    streaming_store(&dstPtr_line1[w*2*B+B], value);
  }
}