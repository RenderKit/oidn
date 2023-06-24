// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "metal_kernel_constants.h"
#include "metal_kernel_common.h"
#include "metal_transfer_function.h"

template<typename TensorDataT, typename ImageDataT>
void output_kernel(device const TensorDataT* src,
                   device ImageDataT* dst,
                   constant ProcessParams* params,
                   uint2 gid [[thread_position_in_grid]])
{
  if (gid.x >= (uint)params->W || gid.y >= (uint)params->H)
  {
    return;
  }
  
  auto transferFunction = TransferFunction<TensorDataT>(params->func, params->normScale);
  
  int offset = (gid.x + gid.y * params->W) * params->C;
  
  for (int c = 0; c < params->C; c++)
  {
    float value = src[offset + c];
    
    value = clamp(nan_to_zero(value), 0.f, POS_MAX);
    
    value = transferFunction.inverse(value);
    
    if (params->snorm)
    {
      value = value * 2.f - 1.f;
      value = max(value, -1.f);
    }
    
    if (!params->hdr)
      value = min(value, 1.f);
    
    value = value * params->outputScale;
    
    dst[offset + c] = value;
  }
}

kernel void output_process(device const void* src,
                           device void* dst,
                           constant ProcessParams* params,
                           uint2 gid [[thread_position_in_grid]])
{
  if (params->inputDataType == KernelDataType::f32 && params->outputDataType == KernelDataType::f32)
  {
    output_kernel<float, float>(castTo<device const void*, device const float*>(src),
                               castTo<device void*, device float*>(dst),
                               params, gid);
  }
  else if (params->inputDataType == KernelDataType::f16 && params->outputDataType == KernelDataType::f32)
  {
    output_kernel<half, float>(castTo<device const void*, device const half*>(src),
                               castTo<device void*, device float*>(dst),
                               params, gid);
  }
  else if (params->inputDataType == KernelDataType::f16 && params->outputDataType == KernelDataType::f16)
  {
    output_kernel<half, half>(castTo<device const void*, device const half*>(src),
                               castTo<device void*, device half*>(dst),
                               params, gid);
  }
  else if (params->inputDataType == KernelDataType::f32 && params->outputDataType == KernelDataType::f16)
  {
    output_kernel<float, half>(castTo<device const void*, device const float*>(src),
                               castTo<device void*, device half*>(dst),
                               params, gid);
  }
}
