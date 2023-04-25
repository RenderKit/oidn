## Copyright 2009-2023 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

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

# Silence false positive "loop not unrolled" warnings
list(APPEND OIDN_SYCL_COMPILE_FLAGS -Wno-pass-failed)
# FIXME: DPCPP issues a warning when WINAPI is used:
# warning: '__stdcall' calling convention is not supported for this target
if(WIN32)
  list(APPEND OIDN_SYCL_COMPILE_FLAGS -Wno-ignored-attributes)
endif()

macro(oidn_add_sycl_library name)
  add_library(${name} SHARED ${ARGN} ${OIDN_RESOURCE})
  set_property(TARGET ${name} PROPERTY VERSION ${PROJECT_VERSION})
  set_property(TARGET ${name} PROPERTY CXX_STANDARD 17)
  target_compile_options(${name} PRIVATE ${OIDN_SYCL_COMPILE_FLAGS})
  target_link_options(${name} PRIVATE ${OIDN_SYCL_LINK_FLAGS})
  target_link_libraries(${name} PRIVATE ${PROJECT_NAME}_core)
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
  oidn_add_sycl_arch_library(${PROJECT_NAME}_device_sycl_xelp sycl_conv_xelp.cpp)
  target_link_options(${PROJECT_NAME}_device_sycl_xelp PRIVATE
    -Xsycl-target-backend=spir64_gen "-device ${OIDN_SYCL_AOT_TARGETS_XELP}"
  )

  # Xe-HPG AOT kernel library
  set(OIDN_SYCL_AOT_TARGETS_XEHPG acm-g10,acm-g11,acm-g12)
  oidn_add_sycl_arch_library(${PROJECT_NAME}_device_sycl_xehpg sycl_conv_xehpg.cpp)
  target_link_options(${PROJECT_NAME}_device_sycl_xehpg PRIVATE
    -Xsycl-target-backend=spir64_gen "-device ${OIDN_SYCL_AOT_TARGETS_XEHPG} -options '-doubleGRF'"
  )

  set(OIDN_SYCL_AOT_TARGETS ${OIDN_SYCL_AOT_TARGETS_XELP},${OIDN_SYCL_AOT_TARGETS_XEHPG})

  if(UNIX)
    # Xe-HPC AOT kernel library
    set(OIDN_SYCL_AOT_TARGETS_XEHPC pvc-sdv,pvc)
    oidn_add_sycl_arch_library(${PROJECT_NAME}_device_sycl_xehpc sycl_conv_xehpc.cpp)
    target_link_options(${PROJECT_NAME}_device_sycl_xehpc PRIVATE
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
oidn_add_sycl_library(${PROJECT_NAME}_device_sycl ${OIDN_SYCL_SOURCES} ${OIDN_GPU_SOURCES})

if(OIDN_DEVICE_SYCL_AOT)
  target_compile_definitions(${PROJECT_NAME}_device_sycl PRIVATE OIDN_DEVICE_SYCL_AOT)

  target_link_options(${PROJECT_NAME}_device_sycl PRIVATE
    -Xsycl-target-backend=spir64_gen "-device ${OIDN_SYCL_AOT_TARGETS}"
  )

  target_link_libraries(${PROJECT_NAME}_device_sycl PRIVATE
    ${PROJECT_NAME}_device_sycl_xelp
    ${PROJECT_NAME}_device_sycl_xehpg
  )
  if(UNIX)
    target_link_libraries(${PROJECT_NAME}_device_sycl PRIVATE
      ${PROJECT_NAME}_device_sycl_xehpc
    )
  endif()
endif()

target_link_libraries(${PROJECT_NAME}_device_sycl PRIVATE LevelZero::LevelZero)
oidn_strip_symbols(${PROJECT_NAME}_device_sycl)