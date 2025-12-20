// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"

struct CPUConvKernel
{
  uniform TensorAccessor3D_ChwBc src;
  uniform TensorAccessor4D_IOhwBiBo weight;
  uniform TensorAccessor1D bias;
  uniform TensorAccessor3D_ChwBc dst;
  uniform bool relu;
};

#define KW 3 // kerned width
#define KH 3 // kerned height
#define PW 1 // padding width on each side
#define PH 1 // padding height on each side

#define blockC programCount

template<typename T, int blockOCB, int blockOW>
inline unmasked void CPUConvKernel_computeBlock(
                       const uniform int8* uniform srcPtr,
                       uniform size_t srcHByteStride,
                       const uniform int8* uniform weightPtr,
                       const uniform int8* uniform biasPtr,
                       uniform int8* uniform dstPtr,
                       uniform size_t dstCByteStride,
                       uniform size_t khEnd,
                       uniform size_t kwBegin, uniform size_t kwEnd,
                       uniform bool relu)
{
  varying T accum[blockOCB][blockOW];

  if (biasPtr)
  {
    #pragma unroll
    for (uniform size_t bocb = 0; bocb < blockOCB; ++bocb)
    {
      #pragma unroll
      for (uniform size_t bow = 0; bow < blockOW; ++bow)
        accum[bocb][bow] = *((const varying T* uniform)biasPtr + bocb);
    }
  }
  else
  {
    #pragma unroll
    for (uniform size_t bocb = 0; bocb < blockOCB; ++bocb)
    {
      #pragma unroll
      for (uniform size_t bow = 0; bow < blockOW; ++bow)
        accum[bocb][bow] = *((const varying T* uniform)(dstPtr + bocb * dstCByteStride) + bow);
    }
  }

  #pragma nounroll
  for (uniform size_t kh = 0; kh < khEnd; ++kh)
  {
    #pragma nounroll
    for (uniform size_t kw = kwBegin; kw < kwEnd; ++kw)
    {
      #pragma unroll
      for (uniform size_t i = 0; i < blockC; ++i)
      {
        #pragma unroll
        for (uniform size_t bocb = 0; bocb < blockOCB; ++bocb)
        {
          const varying T weightVec =
            *((const varying T* uniform)weightPtr + (bocb * KW * KH + kw) * blockC + i);

          #pragma unroll
          for (uniform size_t bow = 0; bow < blockOW; ++bow)
          {
            const varying T srcVec = *((const uniform T* uniform)srcPtr + (bow + kw - PW) * blockC + i);
            accum[bocb][bow] += srcVec * weightVec;
          }
        }
      }
    }

    srcPtr += srcHByteStride;
    weightPtr += KW * blockC * blockC * sizeof(uniform T);
  }

  if (relu)
  {
    #pragma unroll
    for (uniform size_t bocb = 0; bocb < blockOCB; ++bocb)
    {
      #pragma unroll
      for (uniform size_t bow = 0; bow < blockOW; ++bow)
        accum[bocb][bow] = max(accum[bocb][bow], 0);
    }
  }

  #pragma unroll
  for (uniform size_t bocb = 0; bocb < blockOCB; ++bocb)
  {
    #pragma unroll
    for (uniform size_t bow = 0; bow < blockOW; ++bow)
      *((varying T* uniform)(dstPtr + bocb * dstCByteStride) + bow) = accum[bocb][bow];
  }
}

template<typename T, int blockOCB, int blockOW>
unmasked void CPUConvKernel_compute(const uniform CPUConvKernel* uniform self,
                                    uniform int ocb, uniform int oh,
                                    uniform int owBegin, uniform int owEnd)
{
  const uniform int oc = ocb * blockC;

#if KH == 3 && PH == 1
  const uniform int khBegin = oh > 0 ? 0 : 1;
  const uniform int khEnd   = oh < self->dst.H-1 ? 3 : 2;
#else
  const uniform int khBegin = max(PH - oh, 0);
  const uniform int khEnd   = KH - max(PH + oh - (self->dst.H-1), 0);
#endif

  for (uniform int ic = 0; ic < self->src.C; ic += blockC)
  {
    const uniform int8* uniform srcPtr    = Tensor_getPtr<T>(self->src, ic, oh + khBegin - PH, owBegin);
    const uniform int8* uniform weightPtr = Tensor_getPtr<T>(self->weight, oc, ic, khBegin, 0);
    const uniform int8* uniform biasPtr   = (ic == 0) ? Tensor_getPtr<T>(self->bias, oc) : NULL;
    uniform int8* uniform dstPtr          = Tensor_getPtr<T>(self->dst, oc, oh, owBegin);
    const uniform bool relu = self->relu && ic == (self->src.C - blockC);

    uniform int ow = owBegin; // owBegin/owEnd *must* be aligned to block boundaries
    while (ow < owEnd)
    {
      if (ow > PW - 1 && ow + blockOW + PW - 1 < self->dst.W)
      {
        // Fast path (no padding, width blocking)
        CPUConvKernel_computeBlock<T, blockOCB, blockOW>(
          srcPtr, self->src.hByteStride,
          weightPtr, biasPtr,
          dstPtr, self->dst.CByteStride,
          khEnd - khBegin,
          0, KW,
          relu);

        srcPtr += blockOW * blockC * sizeof(uniform T);
        dstPtr += blockOW * blockC * sizeof(uniform T);
        ow += blockOW;
      }
      else
      {
        // Slow path (padding, no width blocking)
        CPUConvKernel_computeBlock<T, blockOCB, 1>(
          srcPtr, self->src.hByteStride,
          weightPtr, biasPtr,
          dstPtr, self->dst.CByteStride,
          khEnd - khBegin,
        #if KW == 3 && PW == 1
          ow > 0 ? 0 : 1,
          ow < self->dst.W-1 ? 3 : 2,
        #else
          max(PW - ow, 0),
          KW - max(PW + ow - (self->dst.W-1), 0),
        #endif
          relu);

        srcPtr += blockC * sizeof(uniform T);
        dstPtr += blockC * sizeof(uniform T);
        ow++;
      }
    }
  }
}

// Kernel variants optimized for different ISAs
#if defined(OIDN_ARCH_AVX512)
  #define maxBlockOCB 4
  #define blockOW1 10
  #define blockOW2 10
  #define blockOW3 7
  #define blockOW4 6
#elif defined(OIDN_ARCH_AVX2)
  #define maxBlockOCB 4
  #define blockOW1 5
  #define blockOW2 5
  #define blockOW3 3
  #define blockOW4 3
#elif defined(OIDN_ARCH_NEON)
  #define maxBlockOCB 3
  #define blockOW1 9
  #define blockOW2 5
  #define blockOW3 3
#elif defined(OIDN_ARCH_SSE4)
  #define maxBlockOCB 1
  #define blockOW1 5
#else
  #error "Unsupported architecture"
#endif

export uniform int CPUConvKernel_getMaxBlockOCB()
{
  return maxBlockOCB;
}

export uniform int CPUConvKernel_getBlockOW(uniform int blockOCB)
{
  switch (blockOCB)
  {
  case 1: return blockOW1;
#if maxBlockOCB >= 2
  case 2: return blockOW2;
#endif
#if maxBlockOCB >= 3
  case 3: return blockOW3;
#endif
#if maxBlockOCB >= 4
  case 4: return blockOW4;
#endif
  default: return 0;
  }
}

export void CPUConvKernel_run_f32(const uniform CPUConvKernel* uniform self,
                                  uniform int blockOCB, uniform int ocb, uniform int oh,
                                  uniform int owBegin, uniform int owEnd)
{
  switch (blockOCB)
  {
  case 1: CPUConvKernel_compute<float, 1, blockOW1>(self, ocb, oh, owBegin, owEnd); break;
#if maxBlockOCB >= 2
  case 2: CPUConvKernel_compute<float, 2, blockOW2>(self, ocb, oh, owBegin, owEnd); break;
#endif
#if maxBlockOCB >= 3
  case 3: CPUConvKernel_compute<float, 3, blockOW3>(self, ocb, oh, owBegin, owEnd); break;
#endif
#if maxBlockOCB >= 4
  case 4: CPUConvKernel_compute<float, 4, blockOW4>(self, ocb, oh, owBegin, owEnd); break;
#endif
  }
}