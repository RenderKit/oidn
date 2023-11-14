// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "../gpu/gpu_autoexposure.h"
#include "../gpu/gpu_input_process.h"
#include "../gpu/gpu_output_process.h"
#include "../gpu/gpu_image_copy.h"

OIDN_NAMESPACE_USING

#define OIDN_DEFINE_BASIC_KERNEL_2D(kernelName, Kernel)                \
  kernel void kernelName(constant Kernel& kernelFunc,                  \
                         uint2 globalID   [[thread_position_in_grid]], \
                         uint2 globalSize [[threads_per_grid]])        \
  {                                                                    \
    WorkItem<2> it(globalID, globalSize);                              \
    const Kernel kernelFuncCopy = kernelFunc;                          \
    kernelFuncCopy(it);                                                \
  }

#define OIDN_DEFINE_GROUP_KERNEL_2D(kernelName, Kernel)                                \
  kernel void kernelName(constant Kernel& kernelFunc,                                  \
                         uint2 globalID   [[thread_position_in_grid]],                 \
                         uint2 globalSize [[threads_per_grid]],                        \
                         uint2 localID    [[thread_position_in_threadgroup]],          \
                         uint2 localSize  [[threads_per_threadgroup]],                 \
                         uint2 groupID    [[threadgroup_position_in_grid]],            \
                         uint2 numGroups  [[threadgroups_per_grid]])                   \
  {                                                                                    \
    WorkGroupItem<2> it(globalID, globalSize, localID, localSize, groupID, numGroups); \
    const Kernel kernelFuncCopy = kernelFunc;                                          \
    kernelFuncCopy(it);                                                                \
  }

#define OIDN_DEFINE_GROUP_LOCAL_KERNEL_1D(kernelName, Kernel)                          \
  kernel void kernelName(constant Kernel& kernelFunc,                                  \
                         uint globalID   [[thread_position_in_grid]],                  \
                         uint globalSize [[threads_per_grid]],                         \
                         uint localID    [[thread_position_in_threadgroup]],           \
                         uint localSize  [[threads_per_threadgroup]],                  \
                         uint groupID    [[threadgroup_position_in_grid]],             \
                         uint numGroups  [[threadgroups_per_grid]])                    \
  {                                                                                    \
    WorkGroupItem<1> it(globalID, globalSize, localID, localSize, groupID, numGroups); \
    threadgroup Kernel::Local local;                                                   \
    const Kernel kernelFuncCopy = kernelFunc;                                          \
    kernelFuncCopy(it, &local);                                                        \
  }

#define OIDN_DEFINE_GROUP_LOCAL_KERNEL_2D(kernelName, Kernel)                          \
  kernel void kernelName(constant Kernel& kernelFunc,                                  \
                         uint2 globalID   [[thread_position_in_grid]],                 \
                         uint2 globalSize [[threads_per_grid]],                        \
                         uint2 localID    [[thread_position_in_threadgroup]],          \
                         uint2 localSize  [[threads_per_threadgroup]],                 \
                         uint2 groupID    [[threadgroup_position_in_grid]],            \
                         uint2 numGroups  [[threadgroups_per_grid]])                   \
  {                                                                                    \
    WorkGroupItem<2> it(globalID, globalSize, localID, localSize, groupID, numGroups); \
    threadgroup Kernel::Local local;                                                   \
    const Kernel kernelFuncCopy = kernelFunc;                                          \
    kernelFuncCopy(it, &local);                                                        \
  }

#define COMMA ,

// -------------------------------------------------------------------------------------------------

OIDN_DEFINE_GROUP_LOCAL_KERNEL_2D(autoexposureDownsample, GPUAutoexposureDownsampleKernel<AutoexposureParams::maxBinSize>)
OIDN_DEFINE_GROUP_LOCAL_KERNEL_1D(autoexposureReduce_1024, GPUAutoexposureReduceKernel<1024>)
OIDN_DEFINE_GROUP_LOCAL_KERNEL_1D(autoexposureReduceFinal_1024, GPUAutoexposureReduceFinalKernel<1024>)

OIDN_DEFINE_GROUP_KERNEL_2D(inputProcess_f16_hwc_3, GPUInputProcessKernel<half COMMA TensorLayout::hwc COMMA 3>)
OIDN_DEFINE_GROUP_KERNEL_2D(inputProcess_f16_hwc_6, GPUInputProcessKernel<half COMMA TensorLayout::hwc COMMA 6>)
OIDN_DEFINE_GROUP_KERNEL_2D(inputProcess_f16_hwc_9, GPUInputProcessKernel<half COMMA TensorLayout::hwc COMMA 9>)

OIDN_DEFINE_BASIC_KERNEL_2D(outputProcess_f16_hwc, GPUOutputProcessKernel<half COMMA TensorLayout::hwc>)

OIDN_DEFINE_BASIC_KERNEL_2D(imageCopy, GPUImageCopyKernel)