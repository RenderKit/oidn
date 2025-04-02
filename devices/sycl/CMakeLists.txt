## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Options
if((CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 18) OR
   (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 2024.2))
   option(OIDN_DEVICE_SYCL_AOT "Enable AOT compilation for SYCL kernels (recommended)." ON)
elseif(OIDN_DEVICE_SYCL_AOT)
  message(FATAL_ERROR "AOT compilation for SYCL kernels requires a newer DPC++ compiler")
endif()

option(OIDN_DEVICE_SYCL_JIT_CACHE "Enable JIT cache for SYCL kernels (recommended)." ON)
mark_as_advanced(OIDN_DEVICE_SYCL_JIT_CACHE)

# Check the generator
if(NOT CMAKE_GENERATOR MATCHES "Ninja" AND NOT CMAKE_GENERATOR MATCHES "Unix Makefiles")
  message(FATAL_ERROR "Building with SYCL support requires Ninja or Make")
endif()

# Check the DPC++ compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-fsycl OIDN_DEVICE_SYCL_SUPPORTED)
  if(NOT OIDN_DEVICE_SYCL_SUPPORTED)
    message(FATAL_ERROR "Building with SYCL support requires oneAPI DPC++ Compiler as the C/C++ compiler")
  endif()

  if(MSVC)
    message(FATAL_ERROR "clang-cl is not supported for SYCL compilation. Please use regular clang instead.")
  endif()

  if(OIDN_DEVICE_SYCL_AOT)
    find_program(_OIDN_DEVICE_SYCL_OCLOC_FOUND NAMES ocloc)
    mark_as_advanced(_OIDN_DEVICE_SYCL_OCLOC_FOUND)

    if(_OIDN_DEVICE_SYCL_OCLOC_FOUND)
      message(STATUS "Found OCLOC: ${_OIDN_DEVICE_SYCL_OCLOC_FOUND}")
    else()
      message(FATAL_ERROR "Building with OIDN_DEVICE_SYCL_AOT requires Intel OpenCL Offline Compiler (OCLOC) to be installed")
    endif()
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
  set(OIDN_ICX_MIN_VERSION 2024.0)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${OIDN_ICX_MIN_VERSION})
    message(FATAL_ERROR "Building with SYCL support requires Intel oneAPI DPC++/C++ Compiler ${OIDN_ICX_MIN_VERSION} or newer")
  endif()
else()
  message(FATAL_ERROR "Building with SYCL support requires oneAPI DPC++ Compiler as the C/C++ compiler")
endif()

find_package(LevelZero REQUIRED)

if(OIDN_DEVICE_SYCL_AOT)
  set(OIDN_SYCL_COMPILE_FLAGS -fsycl)
  set(OIDN_SYCL_LINK_FLAGS -fsycl -Wno-sycl-target) # FIXME: remove -Wno-sycl-target
else()
  set(OIDN_SYCL_COMPILE_FLAGS -fsycl -fsycl-device-code-split=per_kernel)
  set(OIDN_SYCL_LINK_FLAGS ${OIDN_SYCL_COMPILE_FLAGS})
endif()

# FIXME: multi_ptr specialization is deprecated
list(APPEND OIDN_SYCL_COMPILE_FLAGS -Wno-deprecated-declarations)

# FIXME: DPC++ issues a warning when WINAPI is used:
# warning: '__stdcall' calling convention is not supported for this target
if(WIN32)
  list(APPEND OIDN_SYCL_COMPILE_FLAGS -Wno-ignored-attributes)
endif()

set(OIDN_SYCL_SOURCES_COMMON
  sycl_common.h
  sycl_conv_xe.h
  sycl_conv.h
  sycl_device.h
  sycl_device.cpp
  sycl_device_table.h
  sycl_engine.h
  sycl_engine.cpp
  sycl_external_buffer.h
  sycl_external_buffer.cpp
  sycl_module.cpp
)

set(OIDN_SYCL_SOURCES_ARCH
  sycl_conv_xe2.cpp
  sycl_conv_xehpg.cpp
  sycl_conv_xelp.cpp
)

if(UNIX)
  list(APPEND OIDN_SYCL_SOURCES_ARCH
    sycl_conv_xehpc.cpp
  )
endif()

add_library(OpenImageDenoise_device_sycl SHARED
  ${OIDN_SYCL_SOURCES_COMMON}
  ${OIDN_SYCL_SOURCES_ARCH}
  ${OIDN_GPU_SOURCES}
  ${OIDN_RESOURCE_FILE}
)

set_target_properties(OpenImageDenoise_device_sycl PROPERTIES
  OUTPUT_NAME ${OIDN_LIBRARY_NAME}_device_sycl
  CXX_STANDARD 17
)
if(OIDN_LIBRARY_VERSIONED)
  set_target_properties(OpenImageDenoise_device_sycl PROPERTIES VERSION ${PROJECT_VERSION})
endif()

target_compile_options(OpenImageDenoise_device_sycl PRIVATE ${OIDN_SYCL_COMPILE_FLAGS})
target_link_options(OpenImageDenoise_device_sycl PRIVATE ${OIDN_SYCL_LINK_FLAGS})
target_link_libraries(OpenImageDenoise_device_sycl PRIVATE OpenImageDenoise_core)
oidn_install_module(OpenImageDenoise_device_sycl)

if(OIDN_DEVICE_SYCL_AOT)
  target_compile_definitions(OpenImageDenoise_device_sycl PRIVATE OIDN_DEVICE_SYCL_AOT)

  set(OIDN_SYCL_AOT_TARGETS_XELP tgllp,rkl,adl-s,adl-p,adl-n,dg1,mtl-u,mtl-h)
  set(OIDN_SYCL_AOT_TARGETS_XEHPG acm-g10,acm-g11,acm-g12,arl-h)
  set(OIDN_SYCL_AOT_TARGETS_XE2 lnl-m,bmg-g21-a0)
  if(UNIX)
    set(OIDN_SYCL_AOT_TARGETS_XELP ${OIDN_SYCL_AOT_TARGETS_XELP},pvc-vg)
    set(OIDN_SYCL_AOT_TARGETS_XEHPC pvc-sdv,pvc)
  endif()

  set(OIDN_SYCL_AOT_TARGETS
    ${OIDN_SYCL_AOT_TARGETS_XELP},${OIDN_SYCL_AOT_TARGETS_XEHPG},${OIDN_SYCL_AOT_TARGETS_XE2}
  )
  if(OIDN_SYCL_AOT_TARGETS_XEHPC)
    set(OIDN_SYCL_AOT_TARGETS ${OIDN_SYCL_AOT_TARGETS},${OIDN_SYCL_AOT_TARGETS_XEHPC})
  endif()

  macro(oidn_set_sycl_aot_options sources options)
    set(_final_options ${options})
    set(_final_options "${_final_options} --format zebin")
    set_source_files_properties(${sources} PROPERTIES COMPILE_FLAGS
      "-fsycl-targets=spir64_gen -fno-sycl-rdc -Xsycl-target-backend=spir64_gen \"${_final_options}\""
    )
  endmacro()

  oidn_set_sycl_aot_options("${OIDN_SYCL_SOURCES_COMMON}" "-device ${OIDN_SYCL_AOT_TARGETS}")
  oidn_set_sycl_aot_options(sycl_conv_xelp.cpp "-device ${OIDN_SYCL_AOT_TARGETS_XELP}")
  oidn_set_sycl_aot_options(sycl_conv_xehpg.cpp
    "-device ${OIDN_SYCL_AOT_TARGETS_XEHPG} -options '-doubleGRF'")
  if(OIDN_SYCL_AOT_TARGETS_XEHPC)
    oidn_set_sycl_aot_options(sycl_conv_xehpc.cpp "-device ${OIDN_SYCL_AOT_TARGETS_XEHPC}")
  endif()
  oidn_set_sycl_aot_options(sycl_conv_xe2.cpp "-device ${OIDN_SYCL_AOT_TARGETS_XE2}")
endif()

if(OIDN_DEVICE_SYCL_JIT_CACHE)
  target_compile_definitions(OpenImageDenoise_device_sycl PRIVATE OIDN_DEVICE_SYCL_JIT_CACHE)
endif()

target_link_libraries(OpenImageDenoise_device_sycl PRIVATE LevelZero::LevelZero)
oidn_strip_symbols(OpenImageDenoise_device_sycl)

## -------------------------------------------------------------------------------------------------
## Install dependencies
## -------------------------------------------------------------------------------------------------

if(OIDN_INSTALL_DEPENDENCIES)
  get_filename_component(_dpcpp_compiler_dir ${CMAKE_CXX_COMPILER} PATH)

  if(WIN32)
    if(EXISTS "${_dpcpp_compiler_dir}/../bin/pi_level_zero.dll")
      file(GLOB _sycl_deps LIST_DIRECTORIES FALSE
        "${_dpcpp_compiler_dir}/../bin/sycl?.dll"
        "${_dpcpp_compiler_dir}/../bin/pi_level_zero.dll"
        "${_dpcpp_compiler_dir}/../bin/pi_win_proxy_loader.dll"
        "${_dpcpp_compiler_dir}/../bin/win_proxy_loader.dll"    # deprecated
      )
    else()
      file(GLOB _sycl_deps LIST_DIRECTORIES FALSE
        "${_dpcpp_compiler_dir}/../bin/sycl?.dll"
        "${_dpcpp_compiler_dir}/../bin/ur_loader.dll"
        "${_dpcpp_compiler_dir}/../bin/ur_adapter_level_zero.dll"
        "${_dpcpp_compiler_dir}/../bin/ur_win_proxy_loader.dll"
      )
    endif()
  else()
    if(EXISTS "${_dpcpp_compiler_dir}/../lib/libpi_level_zero.so")
      file(GLOB _sycl_deps LIST_DIRECTORIES FALSE
        "${_dpcpp_compiler_dir}/../lib/libsycl.so.?"
        "${_dpcpp_compiler_dir}/../lib/libpi_level_zero.so"
      )
    else()
      file(GLOB _sycl_deps LIST_DIRECTORIES FALSE
        "${_dpcpp_compiler_dir}/../lib/libsycl.so.?"
        "${_dpcpp_compiler_dir}/../lib/libur_loader.so"
        "${_dpcpp_compiler_dir}/../lib/libur_adapter_level_zero.so"
        "${_dpcpp_compiler_dir}/../lib/libumf.so"
      )
    endif()
  endif()

  oidn_install_lib_files(${_sycl_deps})
endif()