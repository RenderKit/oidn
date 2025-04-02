## Copyright 2018 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

find_package(Threads REQUIRED)

# oneTBB
set(OIDN_TBB_MIN_VERSION 2017.0)
if(TBB_ROOT)
  find_package(TBB ${OIDN_TBB_MIN_VERSION} COMPONENTS tbb CONFIG NO_DEFAULT_PATH)
else()
  find_package(TBB ${OIDN_TBB_MIN_VERSION} COMPONENTS tbb CONFIG)
endif()
if(NOT TBB_FOUND)
  # Fall back to our FindTBB.cmake
  unset(TBB_DIR CACHE)
  find_package(TBB ${OIDN_TBB_MIN_VERSION} REQUIRED tbb)
endif()

if(OIDN_ARCH STREQUAL "X64")
  # oneDNN
  option(OIDN_DEVICE_CPU_ONEDNN "Enable oneDNN for CPU device." OFF)
  mark_as_advanced(OIDN_DEVICE_CPU_ONEDNN)
  if(OIDN_DEVICE_CPU_ONEDNN)
    set(OIDN_DNNL ON)
    include(oidn_dnnl)
  endif()
elseif(APPLE AND OIDN_ARCH STREQUAL "ARM64")
  # BNNS
  option(OIDN_DEVICE_CPU_BNNS "Enable BNNS for CPU device." ON)
  mark_as_advanced(OIDN_DEVICE_CPU_BNNS)
  if(OIDN_DEVICE_CPU_BNNS)
    set(OIDN_BNNS ON)
    include(oidn_bnns)
  endif()
endif()

# ISPC
if(OIDN_ARCH STREQUAL "X64")
  set(OIDN_ISPC_TARGET_LIST sse4.1-i32x8;avx2-i32x8;avx512skx-i32x16)
elseif(OIDN_ARCH STREQUAL "ARM64")
  set(OIDN_ISPC_TARGET_LIST neon-i32x8)
endif()
set(OIDN_ISPC_ADDRESSING 64)
include(oidn_ispc)

set(OIDN_CPU_SOURCES
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

set(OIDN_CPU_SOURCES_ISPC
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
  platform.isph
  #platform.ispc
  tensor_accessor.isph
  tile.isph
  vec.isph
)

if(OIDN_DNNL)
  list(APPEND OIDN_CPU_SOURCES
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
  list(APPEND OIDN_CPU_SOURCES
    bnns/bnns_common.h
    bnns/bnns_common.cpp
    bnns/bnns_conv.h
    bnns/bnns_conv.cpp
    bnns/bnns_engine.h
    bnns/bnns_engine.cpp
    bnns/bnns_pool.h
    bnns/bnns_pool.cpp
  )
else()
  list(APPEND OIDN_CPU_SOURCES
    cpu_conv.h
    cpu_conv.cpp
  )

  list(APPEND OIDN_CPU_SOURCES_ISPC
    cpu_conv.ispc
    cpu_conv_compute.isph
    cpu_conv_compute_block.isph
  )
endif()

add_library(OpenImageDenoise_device_cpu ${OIDN_LIB_TYPE} ${OIDN_CPU_SOURCES} ${OIDN_RESOURCE_FILE})

set_target_properties(OpenImageDenoise_device_cpu PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_device_cpu
)
if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_device_cpu PROPERTIES VERSION ${PROJECT_VERSION})
endif()

if(OIDN_DNNL)
  target_compile_definitions(OpenImageDenoise_device_cpu PRIVATE OIDN_DNNL)
  ispc_add_definitions(-DOIDN_DNNL)
  target_link_libraries(OpenImageDenoise_device_cpu PRIVATE dnnl)
elseif(OIDN_BNNS)
  target_compile_definitions(OpenImageDenoise_device_cpu PRIVATE OIDN_BNNS)
  ispc_add_definitions(-DOIDN_BNNS)
  target_link_libraries(OpenImageDenoise_device_cpu PRIVATE "-framework Accelerate")
endif()

ispc_target_add_sources(OpenImageDenoise_device_cpu ${OIDN_CPU_SOURCES_ISPC})
if(OIDN_ARCH STREQUAL "X64")
  set(OIDN_ISPC_TARGET_LIST sse2-i32x4;${OIDN_ISPC_TARGET_LIST}) # required just for ISA detection
endif()
ispc_target_add_sources(OpenImageDenoise_device_cpu platform.ispc)

target_link_libraries(OpenImageDenoise_device_cpu PRIVATE OpenImageDenoise_core ${CMAKE_THREAD_LIBS_INIT} TBB::tbb)

if(OIDN_STATIC_LIB)
  oidn_install_static_module(OpenImageDenoise_device_cpu)
  target_link_libraries(OpenImageDenoise PRIVATE OpenImageDenoise_device_cpu)
else()
  oidn_strip_symbols(OpenImageDenoise_device_cpu)
  oidn_install_module(OpenImageDenoise_device_cpu)
endif()

## -------------------------------------------------------------------------------------------------
## Install dependencies
## -------------------------------------------------------------------------------------------------

if(OIDN_INSTALL_DEPENDENCIES)
  # Install TBB
  oidn_install_imported_lib(TBB::tbb)

  # Install TBB plugins
  if(NOT APPLE)
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
      get_target_property(_tbb_lib TBB::tbb IMPORTED_LOCATION_DEBUG)
      set(_tbb_suffix "_debug")
    else()
      get_target_property(_tbb_lib TBB::tbb IMPORTED_LOCATION_RELEASE)
      set(_tbb_suffix "")
    endif()
    get_filename_component(_tbb_dir "${_tbb_lib}" DIRECTORY)
    if(WIN32)
      file(GLOB _tbb_deps LIST_DIRECTORIES FALSE
        "${_tbb_dir}/tbbbind${_tbb_suffix}\.dll"
        "${_tbb_dir}/tbbbind_?_?${_tbb_suffix}\.dll"
      )
    else()
      file(GLOB _tbb_deps LIST_DIRECTORIES FALSE
        "${_tbb_dir}/libtbbbind${_tbb_suffix}\.so.?"
        "${_tbb_dir}/libtbbbind_?_?${_tbb_suffix}\.so.?"
      )
    endif()
    oidn_install_lib_files(${_tbb_deps})
  endif()
endif()