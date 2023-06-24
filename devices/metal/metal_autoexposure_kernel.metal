// Copyright 2023 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

#include <metal_stdlib>
using namespace metal;

#include "metal_kernel_constants.h"
#include "metal_kernel_common.h"
#include "metal_transfer_function.h"

inline float luminance(float3 c)
{
  return 0.212671f * c.r + 0.715160f * c.g + 0.072169f * c.b;
}

float3 nan_to_zero(float3 c)
{
  return float3(nan_to_zero(c.r), nan_to_zero(c.g), nan_to_zero(c.b));
}

template<typename ImageDataT>
void autoexposure_downsample(const device ImageDataT* src,
                             device float* bins,
                             constant AutoexposureParams* params,
                             threadgroup float* scratch [[threadgroup(0)]],
                             uint2 gid [[ threadgroup_position_in_grid ]],
                             uint2 lid [[ thread_position_in_threadgroup ]],
                             uint2 tpg [[ threads_per_threadgroup ]],
                             uint2 tgpg [[ threadgroups_per_grid ]])
{
  const int beginH = gid.y * params->H / tgpg.y;
  const int beginW = gid.x * params->W / tgpg.x;
  const int endH = (gid.y + 1) * params->H / tgpg.y;
  const int endW = (gid.x + 1) * params->W / tgpg.x;

  const int h = beginH + lid.y;
  const int w = beginW + lid.x;

  const int localId = lid.x + lid.y * tpg.x;
  const int localLinearId = gid.x + gid.y * tgpg.x;
  
  const int groupSize = tpg.x * tpg.y;
  
  float L;
  if (h < endH && w < endW)
  {
    const int idx = (w + h * params->W) * 3;
    float3 clr = float3(src[idx+0], src[idx+1], src[idx+2]);
    clr = clamp(nan_to_zero(clr), 0.f, FLT_MAX); // sanitize
    L = luminance(clr);
  }
  else
    L = 0;
  
  scratch[localId] = L;
  
  for (int i = groupSize / 2; i > 0; i >>= 1)
  {
    if (localId < i)
    {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      scratch[localId] += scratch[localId + i];
    }
  }
  
  threadgroup_barrier(mem_flags::mem_threadgroup);
  
  if (localId == 0)
  {
    const float avgL = scratch[0] / float((endH - beginH) * (endW - beginW));
    bins[localLinearId] = avgL;
  }
}

kernel void autoexposure_downsample(const device void* src,
                                    device float* bins,
                                    constant AutoexposureParams* params,
                                    threadgroup float* scratch [[threadgroup(0)]],
                                    uint2 gid [[ threadgroup_position_in_grid ]],
                                    uint2 lid [[ thread_position_in_threadgroup ]],
                                    uint2 tpg [[ threads_per_threadgroup ]],
                                    uint2 tgpg [[ threadgroups_per_grid ]])
{
  if (params->inputDataType == KernelDataType::f32)
  {
    autoexposure_downsample<float>(castTo<device const void*, device const float*>(src),
                                   bins, params, scratch, gid, lid, tpg, tgpg);
  }
  else if (params->inputDataType == KernelDataType::f16)
  {
    autoexposure_downsample<half>(castTo<device const void*, device const half*>(src),
                                   bins, params, scratch, gid, lid, tpg, tgpg);
  }
}

kernel void autoexposure_reduce(const device float* bins,
                                device float* sums,
                                device float* counts,
                                constant AutoexposureParams* params,
                                uint gid [[ threadgroup_position_in_grid ]],
                                uint lid [[ thread_position_in_threadgroup ]],
                                uint tpg [[ threads_per_threadgroup ]],
                                uint tgpg [[ threadgroups_per_grid ]])
{
  const int numBins = params->numBinsH * params->numBinsW;
  const int size = (numBins + tpg - 1) / tpg;
  const int localId = lid;
  const int offset = localId * size;
  const int groupSize = tpg;
  
  float sum = 0;
  int count = 0;
  for (int i = offset; i < (offset + size); i++)
  {
    const float L = bins[i];
    if (L > eps)
    {
      sum += log2(L);
      count++;
    }
  }
  
  sums[localId] = sum;
  counts[localId] = count;
  
  for (int i = groupSize / 2; i > 0; i >>= 1)
  {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (localId < i)
    {
      sums[localId] += sums[localId + i];
      counts[localId] += counts[localId + i];
    }
  }
}
