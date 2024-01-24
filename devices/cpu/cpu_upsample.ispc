// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"

struct CPUUpsampleKernel
{
  uniform TensorAccessor3D src;
  uniform TensorAccessor3D dst;
};

export void CPUUpsampleKernel_run(const uniform CPUUpsampleKernel* uniform self,
                                  uniform int cb, uniform int h)
{
  const uniform size_t H = (uniform size_t)self->src.H;
  const uniform size_t W = (uniform size_t)self->src.W;

  const uniform size_t offset = (cb*H + h) * (W*B);
  uniform float* const uniform srcPtr_line  = (uniform float* uniform)self->src.ptr + offset;
  uniform float* const uniform dstPtr_line0 = (uniform float* uniform)self->dst.ptr + offset * 4;
  uniform float* const uniform dstPtr_line1 = dstPtr_line0 + W*2*B; // next line

  for (uniform size_t w = 0; w < W; ++w)
  {
    const float value = *((varying float* uniform)&srcPtr_line[w*B]);

    streaming_store(&dstPtr_line0[w*2*B  ], value);
    streaming_store(&dstPtr_line0[w*2*B+B], value);
    streaming_store(&dstPtr_line1[w*2*B  ], value);
    streaming_store(&dstPtr_line1[w*2*B+B], value);
  }
}