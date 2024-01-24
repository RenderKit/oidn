// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor_accessor.isph"

struct CPUConvKernel
{
  uniform TensorAccessor3D src;
  uniform TensorAccessor4D weight;
  uniform TensorAccessor1D bias;
  uniform TensorAccessor3D dst;
  uniform bool relu;
};

#define _CPUConvKernel_compute(T, blockOCB) CPUConvKernel_compute_##T##_##blockOCB
#define CPUConvKernel_compute(T, blockOCB) _CPUConvKernel_compute(T, blockOCB)

#define _CPUConvKernel_computeBlock(T, blockOCB, blockOW) CPUConvKernel_computeBlock_##T##_##blockOCB##_##blockOW
#define CPUConvKernel_computeBlock(T, blockOCB, blockOW) _CPUConvKernel_computeBlock(T, blockOCB, blockOW)

#define T float
#define blockC programCount

#define KW 3 // kerned width
#define KH 3 // kerned height
#define PW 1 // padding width on each side
#define PH 1 // padding height on each side

// Kernel variants optimized for different ISAs
#if defined(ISPC_TARGET_AVX512SKX) || defined(ISPC_TARGET_AVX512SPR)
  #define maxBlockOCB 4
  #define blockOW1 10
  #define blockOW2 10
  #define blockOW3 7
  #define blockOW4 6
#elif defined(ISPC_TARGET_AVX2)
  #define maxBlockOCB 4
  #define blockOW1 5
  #define blockOW2 5
  #define blockOW3 3
  #define blockOW4 3
#elif defined(ISPC_TARGET_NEON)
  #define maxBlockOCB 3
  #define blockOW1 9
  #define blockOW2 5
  #define blockOW3 3
#elif defined(ISPC_TARGET_SSE4) || defined(ISPC_TARGET_SSE2)
  #define maxBlockOCB 1
  #define blockOW1 5
#endif

#if maxBlockOCB >= 1
  #define blockOCB 1
  #define blockOW  1
  #include "cpu_conv_compute_block.isph"
  #undef  blockOW
  #define blockOW  blockOW1
  #include "cpu_conv_compute_block.isph"
  #include "cpu_conv_compute.isph"
  #undef  blockOW
  #undef  blockOCB
#endif

#if maxBlockOCB >= 2
  #define blockOCB 2
  #define blockOW  1
  #include "cpu_conv_compute_block.isph"
  #undef  blockOW
  #define blockOW  blockOW2
  #include "cpu_conv_compute_block.isph"
  #include "cpu_conv_compute.isph"
  #undef  blockOW
  #undef  blockOCB
#endif

#if maxBlockOCB >= 3
  #define blockOCB 3
  #define blockOW  1
  #include "cpu_conv_compute_block.isph"
  #undef  blockOW
  #define blockOW  blockOW3
  #include "cpu_conv_compute_block.isph"
  #include "cpu_conv_compute.isph"
  #undef  blockOW
  #undef  blockOCB
#endif

#if maxBlockOCB >= 4
  #define blockOCB 4
  #define blockOW  1
  #include "cpu_conv_compute_block.isph"
  #undef  blockOW
  #define blockOW  blockOW4
  #include "cpu_conv_compute_block.isph"
  #include "cpu_conv_compute.isph"
  #undef  blockOW
  #undef  blockOCB
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

export void CPUConvKernel_run(const uniform CPUConvKernel* uniform self,
                              uniform int blockOCB, uniform int ocb, uniform int oh,
                              uniform int owBegin, uniform int owEnd)
{
  switch (blockOCB)
  {
  case 1: CPUConvKernel_compute(T, 1)(self, ocb, oh, owBegin, owEnd); break;
#if maxBlockOCB >= 2
  case 2: CPUConvKernel_compute(T, 2)(self, ocb, oh, owBegin, owEnd); break;
#endif
#if maxBlockOCB >= 3
  case 3: CPUConvKernel_compute(T, 3)(self, ocb, oh, owBegin, owEnd); break;
#endif
#if maxBlockOCB >= 4
  case 4: CPUConvKernel_compute(T, 4)(self, ocb, oh, owBegin, owEnd); break;
#endif
  }
}