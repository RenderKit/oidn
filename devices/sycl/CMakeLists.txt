## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

# Options
option(OIDN_DEVICE_SYCL_AOT "Enable AOT compilation for SYCL kernels (recommended)." ON)
if(OIDN_DEVICE_SYCL_AOT)
  option(OIDN_DEVICE_SYCL_JIT_FALLBACK
         "Enable JIT fallback for SYCL kernels when AOT is rejected at runtime." ON)
  mark_as_advanced(OIDN_DEVICE_SYCL_JIT_FALLBACK)

  # FIXME: remove in next release
  if((CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 18) OR
     (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 2024.1))
    option(OIDN_DEVICE_SYCL_AOT_SINGLE_BIN
          "Compile all SYCL AOT kernels into a single binary (requires DPC++ 2023-10-26 or newer)." ON)
    mark_as_advanced(OIDN_DEVICE_SYCL_AOT_SINGLE_BIN)
  endif()
endif()
if(NOT OIDN_DEVICE_SYCL_AOT OR OIDN_DEVICE_SYCL_JIT_FALLBACK)
  option(OIDN_DEVICE_SYCL_JIT_CACHE "Enable JIT cache for SYCL kernels." ON)
  mark_as_advanced(OIDN_DEVICE_SYCL_JIT_CACHE)
endif()

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

if(OIDN_DEVICE_SYCL_AOT)
  if(OIDN_DEVICE_SYCL_AOT_SINGLE_BIN)
    set(OIDN_SYCL_COMPILE_FLAGS -fsycl -fsycl-targets=spir64_gen -fno-sycl-rdc)
    set(OIDN_SYCL_LINK_FLAGS -fsycl -Wno-sycl-target) # FIXME: remove -Wno-sycl-target
  else()
    set(OIDN_SYCL_COMPILE_FLAGS -fsycl -fsycl-targets=spir64_gen)
    set(OIDN_SYCL_LINK_FLAGS ${OIDN_SYCL_COMPILE_FLAGS})
  endif()
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

macro(oidn_add_sycl_library target)
  add_library(${target} SHARED ${ARGN} ${OIDN_RESOURCE_FILE})
  set_property(TARGET ${target} PROPERTY VERSION ${PROJECT_VERSION})
  set_property(TARGET ${target} PROPERTY CXX_STANDARD 17)
  target_compile_options(${target} PRIVATE ${OIDN_SYCL_COMPILE_FLAGS})
  target_link_options(${target} PRIVATE ${OIDN_SYCL_LINK_FLAGS})
  target_link_libraries(${target} PRIVATE OpenImageDenoise_core)
  oidn_install_module(${target})
endmacro()

set(OIDN_SYCL_SOURCES_COMMON
  sycl_common.h
  sycl_conv_xe.h
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

set(OIDN_SYCL_SOURCES_ARCH
  sycl_conv_xehpg.cpp
  sycl_conv_xelp.cpp
)
if(UNIX)
  list(APPEND OIDN_SYCL_SOURCES_ARCH
    sycl_conv_xehpc.cpp
  )
endif()

if(NOT OIDN_DEVICE_SYCL_AOT OR OIDN_DEVICE_SYCL_AOT_SINGLE_BIN)
  oidn_add_sycl_library(OpenImageDenoise_device_sycl
    ${OIDN_SYCL_SOURCES_COMMON}
    ${OIDN_SYCL_SOURCES_ARCH}
    ${OIDN_GPU_SOURCES}
  )
else()
  oidn_add_sycl_library(OpenImageDenoise_device_sycl
    ${OIDN_SYCL_SOURCES_COMMON}
    ${OIDN_GPU_SOURCES}
  )
endif()

if(OIDN_DEVICE_SYCL_AOT)
  target_compile_definitions(OpenImageDenoise_device_sycl PRIVATE OIDN_DEVICE_SYCL_AOT)

  set(OIDN_SYCL_AOT_TARGETS_XELP tgllp,rkl,adl-s,adl-p,adl-n,dg1,mtl-m,mtl-p) # FIXME: add pvc-xt-c0-vg
  set(OIDN_SYCL_AOT_TARGETS_XEHPG acm-g10,acm-g11,acm-g12)
  set(OIDN_SYCL_AOT_TARGETS ${OIDN_SYCL_AOT_TARGETS_XELP},${OIDN_SYCL_AOT_TARGETS_XEHPG})
  if(UNIX)
    set(OIDN_SYCL_AOT_TARGETS_XEHPC pvc-sdv,pvc)
    set(OIDN_SYCL_AOT_TARGETS ${OIDN_SYCL_AOT_TARGETS},${OIDN_SYCL_AOT_TARGETS_XEHPC})
  endif()

  if(OIDN_DEVICE_SYCL_AOT_SINGLE_BIN)
    macro(oidn_set_sycl_aot_options sources options)
      set(_final_options ${options})
      set(_final_options "${_final_options} --format zebin")
      if(NOT OIDN_DEVICE_SYCL_JIT_FALLBACK)
        set(_final_options "${_final_options} -exclude_ir") # do not include SPIR-V
      endif()
      set_source_files_properties(${sources}
        PROPERTIES COMPILE_FLAGS "-Xsycl-target-backend=spir64_gen \"${_final_options}\"")
    endmacro()

    oidn_set_sycl_aot_options("${OIDN_SYCL_SOURCES_COMMON}" "-device ${OIDN_SYCL_AOT_TARGETS}")
    oidn_set_sycl_aot_options(sycl_conv_xelp.cpp "-device ${OIDN_SYCL_AOT_TARGETS_XELP}")
    oidn_set_sycl_aot_options(sycl_conv_xehpg.cpp
      "-device ${OIDN_SYCL_AOT_TARGETS_XEHPG} -options '-doubleGRF'")
    if(UNIX)
      oidn_set_sycl_aot_options(sycl_conv_xehpc.cpp "-device ${OIDN_SYCL_AOT_TARGETS_XEHPC}")
    endif()
  else()
    macro(oidn_add_sycl_arch_library target)
      oidn_add_sycl_library(${target} ${ARGN})
      target_include_directories(${target} PRIVATE ${LevelZero_INCLUDE_DIRS})
      oidn_export_all_symbols(${target})
    endmacro()

    macro(oidn_set_sycl_aot_options target options)
      set(_final_options ${options})
      set(_final_options "${_final_options} --format zebin")
      if(NOT OIDN_DEVICE_SYCL_JIT_FALLBACK)
        set(_final_options "${_final_options} -exclude_ir") # do not include SPIR-V
      endif()
      target_link_options(${target} PRIVATE -Xsycl-target-backend=spir64_gen "${_final_options}")
    endmacro()

    # Common
    oidn_set_sycl_aot_options(OpenImageDenoise_device_sycl "-device ${OIDN_SYCL_AOT_TARGETS}")

    # Xe-LP
    oidn_add_sycl_arch_library(OpenImageDenoise_device_sycl_xelp sycl_conv_xelp.cpp)
    oidn_set_sycl_aot_options(OpenImageDenoise_device_sycl_xelp "-device ${OIDN_SYCL_AOT_TARGETS_XELP}")
    target_link_libraries(OpenImageDenoise_device_sycl PRIVATE OpenImageDenoise_device_sycl_xelp)

    # Xe-HPG
    oidn_add_sycl_arch_library(OpenImageDenoise_device_sycl_xehpg sycl_conv_xehpg.cpp)
    oidn_set_sycl_aot_options(OpenImageDenoise_device_sycl_xehpg
      "-device ${OIDN_SYCL_AOT_TARGETS_XEHPG} -options '-doubleGRF'")
    target_link_libraries(OpenImageDenoise_device_sycl PRIVATE OpenImageDenoise_device_sycl_xehpg)

    if(UNIX)
      # Xe-HPC
      oidn_add_sycl_arch_library(OpenImageDenoise_device_sycl_xehpc sycl_conv_xehpc.cpp)
      oidn_set_sycl_aot_options(OpenImageDenoise_device_sycl_xehpc "-device ${OIDN_SYCL_AOT_TARGETS_XEHPC}")
      target_link_libraries(OpenImageDenoise_device_sycl PRIVATE OpenImageDenoise_device_sycl_xehpc)
    endif()
  endif()
endif()

if(OIDN_DEVICE_SYCL_JIT_CACHE AND (NOT OIDN_DEVICE_SYCL_AOT OR OIDN_DEVICE_SYCL_JIT_FALLBACK))
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
      "${_dpcpp_compiler_dir}/../bin/pi_win_proxy_loader.dll"
      "${_dpcpp_compiler_dir}/../bin/win_proxy_loader.dll"    # deprecated
    )
  else()
    file(GLOB _sycl_deps LIST_DIRECTORIES FALSE
      "${_dpcpp_compiler_dir}/../lib/libsycl.so.?"
      "${_dpcpp_compiler_dir}/../lib/libpi_level_zero.so"
    )
  endif()

  oidn_install_lib_files(${_sycl_deps})
endif()