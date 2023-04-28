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
    get_filename_component(OIDN_CXX_COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
    set(OIDN_DEVICE_CUDA_HOST_COMPILER ${OIDN_CXX_COMPILER_DIR}/clang-cl.exe)
  else()
    set(OIDN_DEVICE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
  endif()

  list(JOIN CMAKE_PREFIX_PATH "|" OIDN_PREFIX_PATH_STR)

  ExternalProject_Add(OpenImageDenoise_device_cuda
    PREFIX cuda
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/devices/cuda
    BINARY_DIR cuda/build
    STAMP_DIR cuda/stamp
    LIST_SEPARATOR |
    CMAKE_CACHE_ARGS
      -DCMAKE_PREFIX_PATH:STRING=${OIDN_PREFIX_PATH_STR}
      -DCMAKE_CXX_COMPILER:FILEPATH=${OIDN_DEVICE_CUDA_HOST_COMPILER}
      -DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/cuda/preinstall
      -DCMAKE_INSTALL_BINDIR:PATH=${CMAKE_INSTALL_BINDIR}
      -DCMAKE_INSTALL_LIBDIR:PATH=${CMAKE_INSTALL_LIBDIR}
      -DOIDN_ROOT_BINARY_DIR:PATH=${OIDN_ROOT_BINARY_DIR}
      -DOIDN_INSTALL_RPATH_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
      -DOIDN_API_NAMESPACE:STRING=${OIDN_API_NAMESPACE}
      -DOIDN_ZIP_MODE:BOOL=${OIDN_ZIP_MODE}
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
      message(FATAL_ERROR "Building with HIP support requires AMD ROCm")
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
    DOC
      "HIP compiler."
  )
  mark_as_advanced(OIDN_DEVICE_HIP_COMPILER)

  # Add ROCm to CMAKE_PREFIX_PATH
  set(OIDN_DEVICE_HIP_PREFIX_PATH CMAKE_PREFIX_PATH)
  list(APPEND OIDN_DEVICE_HIP_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})
  list(JOIN OIDN_DEVICE_HIP_PREFIX_PATH "|" OIDN_DEVICE_HIP_PREFIX_PATH_STR)

  ExternalProject_Add(OpenImageDenoise_device_hip
    PREFIX hip
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/devices/hip
    BINARY_DIR hip/build
    STAMP_DIR hip/stamp
    LIST_SEPARATOR |
    CMAKE_CACHE_ARGS
      -DCMAKE_PREFIX_PATH:STRING=${OIDN_DEVICE_HIP_PREFIX_PATH_STR}
      -DCMAKE_CXX_COMPILER:FILEPATH=${OIDN_DEVICE_HIP_COMPILER}
      -DCMAKE_TOOLCHAIN_FILE:FILEPATH=${CMAKE_TOOLCHAIN_FILE}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_CURRENT_BINARY_DIR}/hip/preinstall
      -DCMAKE_INSTALL_BINDIR:PATH=${CMAKE_INSTALL_BINDIR}
      -DCMAKE_INSTALL_LIBDIR:PATH=${CMAKE_INSTALL_LIBDIR}
      -DOIDN_ROOT_BINARY_DIR:PATH=${OIDN_ROOT_BINARY_DIR}
      -DOIDN_INSTALL_RPATH_PREFIX:PATH=${CMAKE_INSTALL_PREFIX}
      -DOIDN_API_NAMESPACE:STRING=${OIDN_API_NAMESPACE}
      -DOIDN_ZIP_MODE:BOOL=${OIDN_ZIP_MODE}
      -DOIDN_SANITIZER:STRING=${OIDN_SANITIZER}
    BUILD_ALWAYS TRUE
    DEPENDS
      OpenImageDenoise_core
  )

  # Due to limitations of CMake, the module is pre-installed at build time to a temporary location,
  # and then copied to the real install location at install time.
  install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/hip/preinstall/ DESTINATION ".")
endif()