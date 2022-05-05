## Copyright 2009-2022 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(DNNL_VERSION_MAJOR 2)
set(DNNL_VERSION_MINOR 6)
set(DNNL_VERSION_PATCH 0)
set(DNNL_VERSION_HASH  "N/A")

set(DNNL_CPU_RUNTIME "TBB")
set(DNNL_CPU_THREADING_RUNTIME "TBB")
set(DNNL_GPU_RUNTIME "NONE")

option(DNNL_ENABLE_JIT_PROFILING
  "Enable registration of oneDNN kernels that are generated at runtime with
  VTune Amplifier. Without the registrations, VTune Amplifier would report
  data collected inside the kernels as `outside any known module`."
  OFF)
mark_as_advanced(DNNL_ENABLE_JIT_PROFILING)

option(DNNL_ENABLE_ITT_TASKS
  "Enable ITT Tasks tagging feature and tag all primitive execution. VTune
  Amplifier can group profiling results based on those ITT tasks and show
  corresponding timeline information."
  OFF)
mark_as_advanced(DNNL_ENABLE_ITT_TASKS)

option(DNNL_BUILD_EXAMPLES "" OFF)
option(DNNL_BUILD_TESTS "" OFF)
option(DNNL_ENABLE_CONCURRENT_EXEC "" ON)

add_subdirectory(oneDNN)