## Copyright 2009-2020 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

set(DNNL_VERSION_MAJOR 1)
set(DNNL_VERSION_MINOR 6)
set(DNNL_VERSION_PATCH 1)
set(DNNL_VERSION_HASH  "N/A")

set(DNNL_CPU_RUNTIME "TBB")
set(DNNL_CPU_THREADING_RUNTIME "TBB")
set(DNNL_GPU_RUNTIME "NONE")

configure_file(
  "${PROJECT_SOURCE_DIR}/mkl-dnn/include/dnnl_config.h.in"
  "${PROJECT_BINARY_DIR}/mkl-dnn/include/dnnl_config.h"
)
configure_file(
  "${PROJECT_SOURCE_DIR}/mkl-dnn/include/dnnl_version.h.in"
  "${PROJECT_BINARY_DIR}/mkl-dnn/include/dnnl_version.h"
)

file(GLOB_RECURSE DNNL_SOURCES
  mkl-dnn/src/common/*.[ch]
  mkl-dnn/src/common/*.[ch]pp
  mkl-dnn/src/cpu/bfloat16.cpp
  mkl-dnn/src/cpu/cpu_concat.cpp
  mkl-dnn/src/cpu/cpu_concat_pd.hpp
  mkl-dnn/src/cpu/cpu_convolution_list.cpp
  mkl-dnn/src/cpu/cpu_convolution_pd.hpp
  mkl-dnn/src/cpu/cpu_engine.[ch]pp
  mkl-dnn/src/cpu/cpu_memory_storage.hpp
  mkl-dnn/src/cpu/cpu_pooling_list.cpp
  mkl-dnn/src/cpu/cpu_pooling_pd.hpp
  mkl-dnn/src/cpu/cpu_primitive.hpp
  mkl-dnn/src/cpu/cpu_reducer.[ch]pp
  mkl-dnn/src/cpu/cpu_reorder.cpp
  mkl-dnn/src/cpu/cpu_reorder_pd.hpp
  mkl-dnn/src/cpu/cpu_stream.hpp
  mkl-dnn/src/cpu/cpu_sum.cpp
  mkl-dnn/src/cpu/platform.[ch]pp
  mkl-dnn/src/cpu/simple_q10n.hpp
  mkl-dnn/src/cpu/simple_reorder.hpp
  mkl-dnn/src/cpu/wino_reorder.hpp
  mkl-dnn/src/cpu/x64/cpu_barrier.[ch]pp
  mkl-dnn/src/cpu/x64/cpu_isa_traits.[ch]pp
  mkl-dnn/src/cpu/x64/jit_avx2_conv_kernel_f32.[ch]pp
  mkl-dnn/src/cpu/x64/jit_avx2_convolution.[ch]pp
  mkl-dnn/src/cpu/x64/jit_avx512_core_bf16cvt.[ch]pp
  mkl-dnn/src/cpu/x64/jit_avx512_core_f32_wino_conv_*.[ch]pp
  mkl-dnn/src/cpu/x64/jit_generator.hpp
  mkl-dnn/src/cpu/x64/jit_primitive_conf.hpp
  mkl-dnn/src/cpu/x64/jit_sse41_conv_kernel_f32.[ch]pp
  mkl-dnn/src/cpu/x64/jit_sse41_convolution.[ch]pp
  mkl-dnn/src/cpu/x64/jit_transpose_src_utils.[ch]pp
  mkl-dnn/src/cpu/x64/jit_uni_eltwise.[ch]pp
  mkl-dnn/src/cpu/x64/jit_uni_eltwise_injector.[ch]pp
  mkl-dnn/src/cpu/x64/jit_uni_pooling.[ch]pp
  mkl-dnn/src/cpu/x64/jit_uni_pool_kernel.[ch]pp
  mkl-dnn/src/cpu/x64/jit_uni_reorder.[ch]pp
  mkl-dnn/src/cpu/x64/jit_uni_reorder_utils.cpp
  mkl-dnn/src/cpu/x64/jit_utils/*.[ch]
  mkl-dnn/src/cpu/x64/jit_utils/*.[ch]pp
  mkl-dnn/src/cpu/x64/xbyak/*.h
)

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  file(GLOB DNNL_SOURCES_BIGOBJ
    mkl-dnn/src/cpu/cpu_engine.cpp
    mkl-dnn/src/cpu/cpu_reorder.cpp
  )
  set_source_files_properties(${DNNL_SOURCES_BIGOBJ} PROPERTIES COMPILE_FLAGS "/bigobj")
endif()

add_library(dnnl STATIC ${DNNL_SOURCES})

target_include_directories(dnnl
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mkl-dnn/include>
    $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/mkl-dnn/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mkl-dnn/src>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mkl-dnn/src/common>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mkl-dnn/src/cpu>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mkl-dnn/src/cpu/xbyak>
)

target_compile_definitions(dnnl
  PUBLIC
    -DDNNL_ENABLE_CONCURRENT_EXEC
)

set(DNNL_COMPILE_OPTIONS "")
if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # Correct 'jnl' macro/jit issue
  list(APPEND DNNL_COMPILE_OPTIONS "/Qlong-double")
endif()
target_compile_options(dnnl PRIVATE ${DNNL_COMPILE_OPTIONS})

target_link_libraries(dnnl
  PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    TBB
)
if(UNIX AND NOT APPLE)
  # Not every compiler adds -ldl automatically
  target_link_libraries(dnnl PUBLIC ${CMAKE_DL_LIBS})
endif()

if(OIDN_STATIC_LIB)
  install(TARGETS dnnl
    EXPORT
      ${PROJECT_NAME}_Export
    ARCHIVE
      DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  )
endif()