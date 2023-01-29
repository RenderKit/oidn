## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)
find_package(TBB 2017.0 REQUIRED tbb CONFIG)

if(OIDN_ARCH STREQUAL "X64")
  # oneDNN
  set(OIDN_DNNL ON)
  include(oidn_dnnl)
elseif(APPLE AND OIDN_ARCH STREQUAL "ARM64")
  # BNNS
  set(OIDN_BNNS ON)
  include(oidn_bnns)
endif()

# ISPC
if(OIDN_ARCH STREQUAL "X64")
  set(OIDN_ISPC_TARGET_LIST sse4-i32x8;avx2-i32x8;avx512skx-i32x16)
elseif(OIDN_ARCH STREQUAL "ARM64")
  set(OIDN_ISPC_TARGET_LIST neon-i32x8)
endif()
set(OIDN_ISPC_ADDRESSING 64)
include(oidn_ispc)

set(CPU_SOURCES
  cpu_autoexposure.h
  cpu_autoexposure.cpp
  cpu_common.h
  cpu_common.cpp
  cpu_device.h
  cpu_device.cpp
  cpu_engine.h
  cpu_engine.cpp
  cpu_image_copy.h
  cpu_image_copy.cpp
  cpu_input_process.h
  cpu_input_process.cpp
  cpu_module.cpp
  cpu_output_process.h
  cpu_output_process.cpp
  cpu_pool.h
  cpu_pool.cpp
  cpu_upsample.h
  cpu_upsample.cpp
  tasking.h
  tasking.cpp
)

set(CPU_SOURCES_ISPC
  cpu_autoexposure.ispc
  cpu_image_copy.ispc
  cpu_input_process.ispc
  cpu_output_process.ispc
  cpu_pool.ispc
  cpu_upsample.ispc
  color.isph
  color.ispc
  image_accessor.isph
  math.isph
  tensor_accessor.isph
  tile.isph
  vec.isph
)

if(OIDN_DNNL)
  list(APPEND CPU_SOURCES
    dnnl/dnnl_conv.h
    dnnl/dnnl_conv.cpp
    dnnl/dnnl_common.h
    dnnl/dnnl_common.cpp
    dnnl/dnnl_engine.h
    dnnl/dnnl_engine.cpp
    dnnl/dnnl_tensor.h
    dnnl/dnnl_tensor.cpp
  )
elseif(OIDN_BNNS)
  list(APPEND CPU_SOURCES
    bnns/bnns_common.h
    bnns/bnns_common.cpp
    bnns/bnns_conv.h
    bnns/bnns_conv.cpp
    bnns/bnns_engine.h
    bnns/bnns_engine.cpp
    bnns/bnns_pool.h
    bnns/bnns_pool.cpp
  )
endif()

add_library(${PROJECT_NAME}_device_cpu SHARED ${CPU_SOURCES})
set_property(TARGET ${PROJECT_NAME}_device_cpu PROPERTY VERSION ${PROJECT_VERSION})

if(OIDN_DNNL)
  target_compile_definitions(${PROJECT_NAME}_device_cpu PRIVATE OIDN_DNNL)
  ispc_add_definitions(-DOIDN_DNNL)
  target_link_libraries(${PROJECT_NAME}_device_cpu PRIVATE dnnl)
elseif(OIDN_BNNS)
  target_compile_definitions(${PROJECT_NAME}_device_cpu PRIVATE OIDN_BNNS)
  ispc_add_definitions(-DOIDN_BNNS)
  target_link_libraries(${PROJECT_NAME}_device_cpu PRIVATE "-framework Accelerate")
endif()

ispc_target_add_sources(${PROJECT_NAME}_device_cpu ${CPU_SOURCES_ISPC})
target_link_libraries(${PROJECT_NAME}_device_cpu PRIVATE ${PROJECT_NAME}_core ${CMAKE_THREAD_LIBS_INIT} TBB::tbb)
oidn_export_symbols(${PROJECT_NAME}_device_cpu)