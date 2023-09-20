## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Options
option(OIDN_DEVICE_SYCL_AOT "Enable AOT compilation for SYCL kernels (recommended)." ON)
if(OIDN_DEVICE_SYCL_AOT)
  option(OIDN_DEVICE_SYCL_JIT_FALLBACK "Enable JIT fallback for SYCL kernels when AOT is rejected at runtime." ON)
  mark_as_advanced(OIDN_DEVICE_SYCL_JIT_FALLBACK)
else()
  option(OIDN_DEVICE_SYCL_JIT_CACHE "Enable JIT cache for SYCL kernels." ON)
  mark_as_advanced(OIDN_DEVICE_SYCL_JIT_CACHE)
endif()

# Check the generator
# FIXME: Make support is currently broken due to IGC_EnableZEBinary
#if(NOT CMAKE_GENERATOR MATCHES "Ninja" AND NOT CMAKE_GENERATOR MATCHES "Unix Makefiles")
if(NOT CMAKE_GENERATOR MATCHES "Ninja")
  message(FATAL_ERROR "Building with SYCL support requires Ninja")
endif()

# Check the DPC++ compiler
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(-fsycl OIDN_DEVICE_SYCL_SUPPORTED)
  if(NOT OIDN_DEVICE_SYCL_SUPPORTED)
    message(FATAL_ERROR "Building with SYCL support requires oneAPI DPC++ Compiler as the C/C++ compiler")
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
elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  set(OIDN_ICX_MIN_VERSION 2024.0)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${OIDN_ICX_MIN_VERSION})
    message(FATAL_ERROR "Building with SYCL support requires Intel oneAPI DPC++/C++ Compiler ${OIDN_ICX_MIN_VERSION} or newer")
  endif()
else()
  message(FATAL_ERROR "Building with SYCL support requires oneAPI DPC++ Compiler as the C/C++ compiler")
endif()

find_package(LevelZero REQUIRED)

set(OIDN_SYCL_COMPILE_FLAGS -fsycl)

if(OIDN_DEVICE_SYCL_AOT)
  list(APPEND OIDN_SYCL_COMPILE_FLAGS -fsycl-targets=spir64_gen)
else()
  # Minimize JIT compile time
  list(APPEND OIDN_SYCL_COMPILE_FLAGS -fsycl-device-code-split=per_kernel)
endif()

set(OIDN_SYCL_LINK_FLAGS ${OIDN_SYCL_COMPILE_FLAGS})

# FIXME: DPCPP issues a warning when WINAPI is used:
# warning: '__stdcall' calling convention is not supported for this target
if(WIN32)
  list(APPEND OIDN_SYCL_COMPILE_FLAGS -Wno-ignored-attributes)
endif()

macro(oidn_add_sycl_library target)
  add_library(${target} SHARED ${ARGN} ${OIDN_RESOURCE_FILE})
  set_property(TARGET ${target} PROPERTY VERSION ${PROJECT_VERSION})
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
  target_compile_options(${target} PRIVATE ${OIDN_SYCL_COMPILE_FLAGS})
  target_link_options(${target} PRIVATE ${OIDN_SYCL_LINK_FLAGS})
  target_link_libraries(${target} PRIVATE OpenImageDenoise_core)
  oidn_install_module(${target})
endmacro()

set(OIDN_SYCL_SOURCES
  sycl_common.h
  sycl_conv_gen.h
  sycl_conv.h
  sycl_device_ids.h
  sycl_device.h
  sycl_device.cpp
  sycl_engine.h
  sycl_engine.cpp
  sycl_external_buffer.h
  sycl_external_buffer.cpp
  sycl_module.cpp
)

if(OIDN_DEVICE_SYCL_AOT)
  # FIXME: IGC currently ignores "--format zebin" for TGL/ADL, so we need to set an environment variable
  # instead but IGC supports this only on Linux. On Windows we're stuck with the old format.
  # Make support should be re-enabled after removing this workaround.
  if(UNIX)
    set(CMAKE_CXX_LINKER_LAUNCHER "IGC_EnableZEBinary=1" ${CMAKE_CXX_LINKER_LAUNCHER})
  endif()

  macro(oidn_add_sycl_arch_library target)
    oidn_add_sycl_library(${target} ${ARGN})
    target_include_directories(${target} PRIVATE ${LevelZero_INCLUDE_DIRS})
    oidn_export_all_symbols(${target})
  endmacro()

  macro(oidn_set_sycl_aot_options target options)
    set(_final_options ${options})
    # We need to force using the new zebin format because it's not the default for TGL/ADL yet
    set(_final_options "${_final_options} --format zebin")
    if(NOT OIDN_DEVICE_SYCL_JIT_FALLBACK)
      # Do not include SPIRV in the binary
      set(_final_options "${_final_options} -exclude_ir")
    endif()
    target_link_options(${target} PRIVATE -Xsycl-target-backend=spir64_gen "${_final_options}")
  endmacro()

  # Xe-LP AOT kernel library
  set(OIDN_SYCL_AOT_TARGETS_XELP tgllp,rkl,adl-s,adl-p,adl-n,dg1)
  oidn_add_sycl_arch_library(OpenImageDenoise_device_sycl_xelp sycl_conv_xelp.cpp)
  oidn_set_sycl_aot_options(OpenImageDenoise_device_sycl_xelp "-device ${OIDN_SYCL_AOT_TARGETS_XELP}")

  # Xe-HPG AOT kernel library
  set(OIDN_SYCL_AOT_TARGETS_XEHPG acm-g10,acm-g11,acm-g12)
  oidn_add_sycl_arch_library(OpenImageDenoise_device_sycl_xehpg sycl_conv_xehpg.cpp)
  oidn_set_sycl_aot_options(OpenImageDenoise_device_sycl_xehpg
    "-device ${OIDN_SYCL_AOT_TARGETS_XEHPG} -options '-doubleGRF'"
  )

  set(OIDN_SYCL_AOT_TARGETS ${OIDN_SYCL_AOT_TARGETS_XELP},${OIDN_SYCL_AOT_TARGETS_XEHPG})

  if(UNIX)
    # Xe-HPC AOT kernel library
    set(OIDN_SYCL_AOT_TARGETS_XEHPC pvc-sdv,pvc)
    oidn_add_sycl_arch_library(OpenImageDenoise_device_sycl_xehpc sycl_conv_xehpc.cpp)
    oidn_set_sycl_aot_options(OpenImageDenoise_device_sycl_xehpc "-device ${OIDN_SYCL_AOT_TARGETS_XEHPC}")
    set(OIDN_SYCL_AOT_TARGETS ${OIDN_SYCL_AOT_TARGETS},${OIDN_SYCL_AOT_TARGETS_XEHPC})
  endif()
else()
  # JIT
  list(APPEND OIDN_SYCL_SOURCES
    sycl_conv_xelp.cpp
    sycl_conv_xehpg.cpp
  )
  if(UNIX)
    list(APPEND OIDN_SYCL_SOURCES
      sycl_conv_xehpc.cpp
    )
  endif()
endif()

# Main SYCL library
oidn_add_sycl_library(OpenImageDenoise_device_sycl ${OIDN_SYCL_SOURCES} ${OIDN_GPU_SOURCES})

if(OIDN_DEVICE_SYCL_AOT)
  target_compile_definitions(OpenImageDenoise_device_sycl PRIVATE OIDN_DEVICE_SYCL_AOT)
  oidn_set_sycl_aot_options(OpenImageDenoise_device_sycl "-device ${OIDN_SYCL_AOT_TARGETS}")

  target_link_libraries(OpenImageDenoise_device_sycl PRIVATE
    OpenImageDenoise_device_sycl_xelp
    OpenImageDenoise_device_sycl_xehpg
  )
  if(UNIX)
    target_link_libraries(OpenImageDenoise_device_sycl PRIVATE
      OpenImageDenoise_device_sycl_xehpc
    )
  endif()
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
    file(GLOB _sycl_deps LIST_DIRECTORIES FALSE
      "${_dpcpp_compiler_dir}/../bin/sycl?.dll"
      "${_dpcpp_compiler_dir}/../bin/pi_level_zero.dll"
      "${_dpcpp_compiler_dir}/../bin/win_proxy_loader.dll"
    )
  else()
    file(GLOB _sycl_deps LIST_DIRECTORIES FALSE
      "${_dpcpp_compiler_dir}/../lib/libsycl.so.?"
      "${_dpcpp_compiler_dir}/../lib/libpi_level_zero.so"
    )
  endif()

  oidn_install_lib_files(${_sycl_deps})
endif()