// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "metal_kernel_constants.h"
#include "metal_kernel_common.h"
#include "metal_transfer_function.h"

template<typename ImageDataT, typename TensorDataT>
void input_kernel(device const ImageDataT* color,
                  device const ImageDataT* albedo,
                  device const ImageDataT* normal,
                  device TensorDataT* dst,
                  constant ProcessParams* params,
                  uint2 gid [[thread_position_in_grid]])
{
  auto transferFunc = TransferFunction<ImageDataT>(params->func, params->normScale);

  if (gid.x >= (uint)params->W || gid.y >= (uint)params->H)
  {
    return;
  }

  int offsetDst = (gid.x + gid.y * params->W) * params->C;
  int offsetSrc = (gid.x + gid.y * params->W) * 3;
  int channel = 0;

  uint h = gid.y - params->tile.hDstBegin;
  uint w = gid.x - params->tile.wDstBegin;

  if ((h < 0 || gid.y >= (uint)params->tile.H) ||
      (w < 0 || gid.x >= (uint)params->tile.W))
  {
    for (int c = 0; c < params->C; c++)
    {
      int channel = 0;
      if (params->color)
      {
        dst[offsetSrc + channel + c] = 0;
        color += 3;
      }
      if (params->albedo)
      {
        dst[offsetSrc + channel + c] = 0;
        color += 3;
      }
      if (params->normal)
        dst[offsetSrc + channel + c] = 0;
    }
    return;
  }

  const float inputScale = params->inputScalePtr ? *params->inputScalePtr : params->inputScale;

  if (params->color)
  {
    for (int c = 0; c < params->C; c++)
    {
      TensorDataT value = color[offsetSrc + c];

      // Scale
      value = value * inputScale;

      // Sanitize
      value = clamp(nan_to_zero(value), params->snorm ? -1.f : 0.f, params->hdr ? POS_MAX : 1.f);

      // Transform to [0..1]
      if (params->snorm)
        value = value * 0.5f + 0.5f;

      // Apply the transfer function
      value = transferFunc.forward(value);

      dst[offsetDst + channel + c] = value;
    }

    channel += 3;
  }

  if (params->albedo)
  {
    for (int c = 0; c < 3; c++)
    {
      TensorDataT value = albedo[offsetSrc + c];

      // Scale
      if (!params->color)
        value = value * inputScale;

      // Sanitize
      value = clamp(nan_to_zero(value), 0.f, 1.f);

      // Apply the transfer function
      if (!params->color)
        value = transferFunc.forward(value);

      // Store
      dst[offsetDst + channel + c] = value;
    }

    channel += 3;
  }

  if (params->normal)
  {
    for (int c = 0; c < 3; c++)
    {
      TensorDataT value = normal[offsetSrc + c];

      // Scale
      if (!params->color)
        value = value * inputScale;

      // Sanitize
      value = clamp(nan_to_zero(value), -1.f, 1.f);

      // Transform to [0..1]
      value = value * 0.5f + 0.5f;

      // Store
      dst[offsetDst + channel + c] = value;
    }
  }
}

kernel void input_process(device const void* color,
                          device const void* albedo,
                          device const void* normal,
                          device void* dst,
                          constant ProcessParams* params,
                          uint2 index [[thread_position_in_grid]])
{
  if (params->inputDataType == KernelDataType::f32 && params->outputDataType == KernelDataType::f32)
  {
    input_kernel<float, float>(castTo<device const void*, device const float*>(color),
                               castTo<device const void*, device const float*>(albedo),
                               castTo<device const void*, device const float*>(normal),
                               castTo<device void*, device float*>(dst),
                               params, index);
  }
  else if (params->inputDataType == KernelDataType::f16 && params->outputDataType == KernelDataType::f32)
  {
    input_kernel<half, float>(castTo<device const void*, device const half*>(color),
                               castTo<device const void*, device const half*>(albedo),
                               castTo<device const void*, device const half*>(normal),
                               castTo<device void*, device float*>(dst),
                               params, index);
  }
  else if (params->inputDataType == KernelDataType::f16 && params->outputDataType == KernelDataType::f16)
  {
    input_kernel<half, half>(castTo<device const void*, device const half*>(color),
                               castTo<device const void*, device const half*>(albedo),
                               castTo<device const void*, device const half*>(normal),
                               castTo<device void*, device half*>(dst),
                               params, index);
  }
  else if (params->inputDataType == KernelDataType::f32 && params->outputDataType == KernelDataType::f16)
  {
    input_kernel<float, half>(castTo<device const void*, device const float*>(color),
                               castTo<device const void*, device const float*>(albedo),
                               castTo<device const void*, device const float*>(normal),
                               castTo<device void*, device half*>(dst),
                               params, index);
  }
}
