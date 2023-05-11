## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

include(ExternalProject)

set(OIDN_GPU_SOURCES
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_autoexposure.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_image_copy.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_input_process.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_output_process.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_pool.h
  ${CMAKE_CURRENT_SOURCE_DIR}/gpu/gpu_upsample.h
)

if(OIDN_DEVICE_CPU)
  add_subdirectory(cpu)
endif()

if(OIDN_DEVICE_SYCL)
  add_subdirectory(sycl)
endif()

if(OIDN_DEVICE_CUDA)
  # We need to build this an external project because we might need to switch to clang-cl
  # CMake does not support GNU-like compilers for CUDA
  if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT MSVC)
    get_filename_component(_cxx_compiler_dir ${CMAKE_CXX_COMPILER} DIRECTORY)
    set(_host_compiler ${_cxx_compiler_dir}/clang-cl.exe)
  else()
    set(_host_compiler ${CMAKE_CXX_COMPILER})
  endif()

  list(JOIN CMAKE_PREFIX_PATH "|" _prefix_path_str)

  ExternalProject_Add(OpenImageDenoise_device_cuda
    PREFIX cuda
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/devices/cuda
    BINARY_DIR cuda/build
    STAMP_DIR cuda/stamp
    LIST_SEPARATOR |
    CMAKE_CACHE_ARGS
      -DCMAKE_PREFIX_PATH:STRING=${_prefix_path_str}
      -DCMAKE_CXX_COMPILER:FILEPATH=${_host_compiler}
      -DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/cuda/preinstall
      -DCMAKE_INSTALL_BINDIR:PATH=${CMAKE_INSTALL_BINDIR}
      -DCMAKE_INSTALL_LIBDIR:PATH=${CMAKE_INSTALL_LIBDIR}
      -DCUDAToolkit_ROOT:PATH=${CUDAToolkit_ROOT}
      -DOIDN_ROOT_BINARY_DIR:PATH=${OIDN_ROOT_BINARY_DIR}
      -DOIDN_INSTALL_RPATH_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
      -DOIDN_INSTALL_DEPENDENCIES:BOOL=${OIDN_INSTALL_DEPENDENCIES}
      -DOIDN_ZIP_MODE:BOOL=${OIDN_ZIP_MODE}
      -DOIDN_API_NAMESPACE:STRING=${OIDN_API_NAMESPACE}
      -DOIDN_SANITIZER:STRING=${OIDN_SANITIZER}
    BUILD_ALWAYS TRUE
    DEPENDS
      OpenImageDenoise_core
  )

  # Due to limitations of CMake, the module is pre-installed at build time to a temporary location,
  # and then copied to the real install location at install time.
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cuda/preinstall/ DESTINATION ".")
endif()

if(OIDN_DEVICE_HIP)
  # Find ROCm
  if(NOT ROCM_PATH)
    find_path(ROCM_PATH
      NAMES
        bin/hipconfig
      HINTS
        $ENV{ROCM_PATH}
      PATHS
        /opt/rocm
      NO_DEFAULT_PATH
      DOC
        "ROCm installation path."
    )
    mark_as_advanced(ROCM_PATH)

    if(ROCM_PATH)
      message(STATUS "Found ROCm: ${ROCM_PATH}")
    else()
      message(FATAL_ERROR "Failed to find ROCm.\nBuilding with HIP support requires ROCm. Please set the ROCM_PATH variable.")
    endif()
  endif()

  # Find HIP compiler
  find_program(OIDN_DEVICE_HIP_COMPILER
    NAMES
      clang++
    HINTS
      ${ROCM_PATH}
    PATH_SUFFIXES
      bin
      llvm/bin
    NO_DEFAULT_PATH
    REQUIRED
    DOC
      "HIP compiler."
  )
  mark_as_advanced(OIDN_DEVICE_HIP_COMPILER)

  # Add ROCm to CMAKE_PREFIX_PATH
  set(_hip_prefix_path CMAKE_PREFIX_PATH)
  list(APPEND _hip_prefix_path ${ROCM_PATH}/hip ${ROCM_PATH})
  list(JOIN _hip_prefix_path "|" _hip_prefix_path_str)

  ExternalProject_Add(OpenImageDenoise_device_hip
    PREFIX hip
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/devices/hip
    BINARY_DIR hip/build
    STAMP_DIR hip/stamp
    LIST_SEPARATOR |
    CMAKE_CACHE_ARGS
      -DCMAKE_PREFIX_PATH:STRING=${_hip_prefix_path_str}
      -DCMAKE_CXX_COMPILER:FILEPATH=${OIDN_DEVICE_HIP_COMPILER}
      -DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/hip/preinstall
      -DCMAKE_INSTALL_BINDIR:PATH=${CMAKE_INSTALL_BINDIR}
      -DCMAKE_INSTALL_LIBDIR:PATH=${CMAKE_INSTALL_LIBDIR}
      -DOIDN_ROOT_BINARY_DIR:PATH=${OIDN_ROOT_BINARY_DIR}
      -DOIDN_INSTALL_RPATH_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
      -DOIDN_INSTALL_DEPENDENCIES:BOOL=${OIDN_INSTALL_DEPENDENCIES}
      -DOIDN_ZIP_MODE:BOOL=${OIDN_ZIP_MODE}
      -DOIDN_API_NAMESPACE:STRING=${OIDN_API_NAMESPACE}
      -DOIDN_SANITIZER:STRING=${OIDN_SANITIZER}
    BUILD_ALWAYS TRUE
    DEPENDS
      OpenImageDenoise_core
  )

  # Due to limitations of CMake, the module is pre-installed at build time to a temporary location,
  # and then copied to the real install location at install time.
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hip/preinstall/ DESTINATION ".")
endif()