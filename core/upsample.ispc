// =============================================================================
// Copyright 2009-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "memory.ih"

struct Upsample
{
  uniform Memory src;
  uniform Memory dst;
};

export void Upsample_kernel(uniform Upsample* uniform self, uniform int ck, uniform int h)
{
  const uniform size_t H = (size_t)self->src.H;
  const uniform size_t W = (size_t)self->src.W;

  const uniform size_t offset = (ck*H + h) * (W*K);
  uniform float* const uniform srcPtr_line  = self->src.ptr + offset;
  uniform float* const uniform dstPtr_line0 = self->dst.ptr + offset * 4;
  uniform float* const uniform dstPtr_line1 = dstPtr_line0 + W*2*K; // next line

  for (uniform size_t w = 0; w < W; ++w)
  {
    // Load vector
    const float value = *((varying float* uniform)&srcPtr_line[w*K]);

    // Store vector 2x2
    streaming_store(&dstPtr_line0[w*2*K  ], value);
    streaming_store(&dstPtr_line0[w*2*K+K], value);
    streaming_store(&dstPtr_line1[w*2*K  ], value);
    streaming_store(&dstPtr_line1[w*2*K+K], value);
  }
}