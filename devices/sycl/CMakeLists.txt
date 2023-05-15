## Copyright 2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

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

    if(NOT _OIDN_DEVICE_SYCL_OCLOC_FOUND)
      message(FATAL_ERROR "Building with OIDN_DEVICE_SYCL_AOT requires Intel OpenCL Offline Compiler (OCLOC) to be installed")
    endif()
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  set(OIDN_ICX_MIN_VERSION 2023.1)
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${OIDN_ICX_MIN_VERSION})
    message(FATAL_ERROR "Building with SYCL support requires Intel oneAPI DPC++/C++ Compiler ${OIDN_ICX_MIN_VERSION} or newer")
  endif()
else()
  message(FATAL_ERROR "Building with SYCL support requires oneAPI DPC++ Compiler as the C/C++ compiler")
endif()

find_package(LevelZero REQUIRED)

option(OIDN_DEVICE_SYCL_AOT "Enable AOT compilation for SYCL kernels (recommended)." ON)

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

macro(oidn_add_sycl_library name)
  add_library(${name} SHARED ${ARGN} ${OIDN_RESOURCE_FILE})
  set_property(TARGET ${name} PROPERTY VERSION ${PROJECT_VERSION})
  set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)
  target_compile_options(${name} PRIVATE ${OIDN_SYCL_COMPILE_FLAGS})
  target_link_options(${name} PRIVATE ${OIDN_SYCL_LINK_FLAGS})
  target_link_libraries(${name} PRIVATE OpenImageDenoise_core)
  oidn_install_module(${name})
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
  macro(oidn_add_sycl_arch_library name)
    oidn_add_sycl_library(${name} ${ARGN})
    target_include_directories(${name} PRIVATE ${LevelZero_INCLUDE_DIRS})
    oidn_export_all_symbols(${name})
  endmacro()

  # Xe-LP AOT kernel library
  set(OIDN_SYCL_AOT_TARGETS_XELP tgllp,rkl,adl-s,adl-p,adl-n,dg1)
  oidn_add_sycl_arch_library(OpenImageDenoise_device_sycl_xelp sycl_conv_xelp.cpp)
  target_link_options(OpenImageDenoise_device_sycl_xelp PRIVATE
    -Xsycl-target-backend=spir64_gen "-device ${OIDN_SYCL_AOT_TARGETS_XELP}"
  )

  # Xe-HPG AOT kernel library
  set(OIDN_SYCL_AOT_TARGETS_XEHPG acm-g10,acm-g11,acm-g12)
  oidn_add_sycl_arch_library(OpenImageDenoise_device_sycl_xehpg sycl_conv_xehpg.cpp)
  target_link_options(OpenImageDenoise_device_sycl_xehpg PRIVATE
    -Xsycl-target-backend=spir64_gen "-device ${OIDN_SYCL_AOT_TARGETS_XEHPG} -options '-doubleGRF'"
  )

  set(OIDN_SYCL_AOT_TARGETS ${OIDN_SYCL_AOT_TARGETS_XELP},${OIDN_SYCL_AOT_TARGETS_XEHPG})

  if(UNIX)
    # Xe-HPC AOT kernel library
    set(OIDN_SYCL_AOT_TARGETS_XEHPC pvc-sdv,pvc)
    oidn_add_sycl_arch_library(OpenImageDenoise_device_sycl_xehpc sycl_conv_xehpc.cpp)
    target_link_options(OpenImageDenoise_device_sycl_xehpc PRIVATE
      -Xsycl-target-backend=spir64_gen "-device ${OIDN_SYCL_AOT_TARGETS_XEHPC}"
    )

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

  target_link_options(OpenImageDenoise_device_sycl PRIVATE
    -Xsycl-target-backend=spir64_gen "-device ${OIDN_SYCL_AOT_TARGETS}"
  )

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