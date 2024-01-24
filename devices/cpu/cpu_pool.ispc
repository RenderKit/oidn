// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"

struct CPUPoolKernel
{
  uniform TensorAccessor3D src;
  uniform TensorAccessor3D dst;
};

export void CPUPoolKernel_run(const uniform CPUPoolKernel* uniform self,
                              uniform int cb, uniform int h)
{
  const uniform size_t H = (uniform size_t)self->dst.H;
  const uniform size_t W = (uniform size_t)self->dst.W;

  const uniform size_t offset = (cb*H + h) * (W*B);
  uniform float* const uniform dstPtr_line  = (uniform float* uniform)self->dst.ptr + offset;
  uniform float* const uniform srcPtr_line0 = (uniform float* uniform)self->src.ptr + offset * 4;
  uniform float* const uniform srcPtr_line1 = srcPtr_line0 + W*2*B; // next line

  for (uniform size_t w = 0; w < W; ++w)
  {
    const float value0 = *((varying float* uniform)&srcPtr_line0[w*2*B  ]);
    const float value1 = *((varying float* uniform)&srcPtr_line0[w*2*B+B]);
    const float value2 = *((varying float* uniform)&srcPtr_line1[w*2*B  ]);
    const float value3 = *((varying float* uniform)&srcPtr_line1[w*2*B+B]);

    const float value = max(max(value0, value1), max(value2, value3));
    streaming_store(&dstPtr_line[w*B], value);
  }
}