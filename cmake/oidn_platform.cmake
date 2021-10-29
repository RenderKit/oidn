## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

include(CheckCXXCompilerFlag)

# Detect the processor architecture
if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Intel(R) Open Image Denoise supports 64-bit platforms only")
endif()
if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64|aarch64")
  set(OIDN_ARCH "ARM64")
else()
  set(OIDN_ARCH "X64")
endif()

# Initialize the compile flags
set(OIDN_C_CXX_FLAGS)
set(OIDN_C_CXX_FLAGS_RELEASE)
set(OIDN_C_CXX_FLAGS_DEBUG)
set(OIDN_CXX_FLAGS)

# UINT8_MAX-like macros are a part of the C99 standard and not a part of the
# C++ standard (see C99 standard 7.18.2 and 7.18.4)
add_definitions(-D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS)

if(MSVC)
  if(CMAKE_BASE_NAME STREQUAL "icx" OR CMAKE_BASE_NAME STREQUAL "icpx")
    # Default fp-model in icx and dpcpp (unlike clang) may be precise or fast=1 depending on the version
    append(OIDN_C_CXX_FLAGS "/fp:precise")
  endif()
  append_if(OIDN_WARN_AS_ERRORS OIDN_C_CXX_FLAGS "/WX")
  # Enable intrinsic functions
  append(OIDN_C_CXX_FLAGS "/Oi")
  # Enable full optimizations
  append(OIDN_C_CXX_FLAGS_RELEASE "/Ox")
  # Package individual functions
  append(OIDN_C_CXX_FLAGS_RELEASE "/Gy")
  # Disable secure warnings
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    append(OIDN_C_CXX_FLAGS "/MP")
    # Disable warning: int -> bool
    append(OIDN_C_CXX_FLAGS "/wd4800")
    # Disable warning: unknown pragma
    append(OIDN_C_CXX_FLAGS "/wd4068")
    # Disable warning: double -> float
    append(OIDN_C_CXX_FLAGS "/wd4305")
    # Disable warning: UNUSED(func)
    append(OIDN_C_CXX_FLAGS "/wd4551")
    # Disable warning: int64_t -> int (tent)
    append(OIDN_C_CXX_FLAGS "/wd4244")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append(OIDN_C_CXX_FLAGS "/MP")
    # Disable warning: option '/Qstd=c++11' is not valid for C compilations (CMake bug?)
    append(OIDN_C_CXX_FLAGS "/Qwd10370")
    # Disable diagnostic: loop was not vectorized with "simd"
    append(OIDN_C_CXX_FLAGS "-Qdiag-disable:13379")
    append(OIDN_C_CXX_FLAGS "-Qdiag-disable:15552")
    append(OIDN_C_CXX_FLAGS "-Qdiag-disable:15335")
    # Disable diagnostic: unknown pragma
    append(OIDN_C_CXX_FLAGS "-Qdiag-disable:3180")
    # Disable diagnostic: foo has been targeted for automatic cpu dispatch
    append(OIDN_C_CXX_FLAGS "-Qdiag-disable:15009")
    # Disable diagnostic: disabling user-directed function packaging (COMDATs)
    append(OIDN_C_CXX_FLAGS "-Qdiag-disable:11031")
    # disable: decorated name length exceeded, name was truncated
    append(OIDN_C_CXX_FLAGS "-Qdiag-disable:2586")
    # disable: disabling optimization; runtime debug checks enabled
    append(OIDN_C_CXX_FLAGS_DEBUG "-Qdiag-disable:10182")
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Disable warning: cannot vectorize some loops with #pragma omp simd
    append(OIDN_C_CXX_FLAGS "-Wno-pass-failed")
    # Disable warning: function is not needed and will not be emitted
    append(OIDN_C_CXX_FLAGS "-Wno-unneeded-internal-declaration")
  endif()
elseif(UNIX OR MINGW)
  if(CMAKE_BASE_NAME STREQUAL "icx" OR CMAKE_BASE_NAME STREQUAL "icpx")
    # Default fp-model in icx and dpcpp (unlike clang) may be precise or fast=1 depending on the version
    append(OIDN_C_CXX_FLAGS "-ffp-model=precise -fno-reciprocal-math")
  endif()
  append(OIDN_C_CXX_FLAGS "-Wall -Wno-unknown-pragmas")
  append_if(OIDN_WARN_AS_ERRORS OIDN_C_CXX_FLAGS "-Werror")
  append(OIDN_C_CXX_FLAGS "-fvisibility=internal")
  append(OIDN_CXX_FLAGS "-fvisibility-inlines-hidden")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Disable warning: cannot vectorize some loops with #pragma omp simd
    append(OIDN_C_CXX_FLAGS "-Wno-pass-failed")
    # Disable warning: function is not needed and will not be emitted
    append(OIDN_C_CXX_FLAGS "-Wno-unneeded-internal-declaration")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Suppress warning on assumptions made regarding overflow (#146)
    append(OIDN_C_CXX_FLAGS "-Wno-strict-overflow")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    # Disable optimizations in debug mode
    append(OIDN_C_CXX_FLAGS_DEBUG "-O0")
    # Workaround for ICC that produces error caused by pragma omp simd collapse(..)
    append(OIDN_C_CXX_FLAGS "-diag-disable:13379")
    append(OIDN_C_CXX_FLAGS "-diag-disable:15552")
    # Disable `was not vectorized: vectorization seems inefficient` remark
    append(OIDN_C_CXX_FLAGS "-diag-disable:15335")
    # Disable diagnostic: foo has been targeted for automatic cpu dispatch
    append(OIDN_C_CXX_FLAGS "-diag-disable:15009")
  endif()
endif()

if(WIN32)
  add_definitions(-D_WIN)
  add_definitions(-DNOMINMAX)
  if(MSVC)
    option(OIDN_STATIC_RUNTIME "Use the static version of the C/C++ runtime library." OFF)
    mark_as_advanced(OIDN_STATIC_RUNTIME)
    if(OIDN_STATIC_RUNTIME)
      foreach(FLAGS
        CMAKE_CXX_FLAGS_DEBUG
        CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS_DEBUG
        CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_RELWITHDEBINFO
      )
        string(REPLACE "/MD" "/MT" ${FLAGS} ${${FLAGS}})
      endforeach()
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
      # Use the default math library instead of libmmd[d]
      string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG " /nodefaultlib:libmmdd.lib")
      string(APPEND CMAKE_EXE_LINKER_FLAGS_RELEASE " /nodefaultlib:libmmd.lib")
      string(APPEND CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO " /nodefaultlib:libmmd.lib")
      string(APPEND CMAKE_SHARED_LINKER_FLAGS_DEBUG " /nodefaultlib:libmmdd.lib")
      string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELEASE " /nodefaultlib:libmmd.lib")
      string(APPEND CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO " /nodefaultlib:libmmd.lib")

      # Link the static version of SVML
      string(APPEND CMAKE_EXE_LINKER_FLAGS " /defaultlib:svml_dispmt.lib")
      string(APPEND CMAKE_SHARED_LINKER_FLAGS " /defaultlib:svml_dispmt.lib")
    endif()
  endif()
endif()

if((UNIX OR MINGW) AND CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  # Link Intel libraries statically
  append(CMAKE_SHARED_LINKER_FLAGS "-static-intel")
  # Tell linker to not complain about missing static libraries
  append(CMAKE_SHARED_LINKER_FLAGS "-diag-disable:10237")
endif()

if(APPLE)
  # Make sure code runs on older macOS versions
  set(CMAKE_OSX_DEPLOYMENT_TARGET 10.11)
  # Link against libc++ which supports C++11 features
  append(OIDN_CXX_FLAGS "-stdlib=libc++")
endif()

## -----------------------------------------------------------------------------
## OpenMP SIMD
## -----------------------------------------------------------------------------

if(WIN32)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options(/Qpar)
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append(OIDN_C_CXX_FLAGS "/Qopenmp-simd")
  endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
  append(OIDN_C_CXX_FLAGS "-qopenmp-simd")
else()
  check_cxx_compiler_flag(-fopenmp-simd compiler-openmp-simd-support)
  if(compiler-openmp-simd-support)
    append(OIDN_C_CXX_FLAGS "-fopenmp-simd")
  endif()
endif()

## -----------------------------------------------------------------------------
## Secure Development Lifecycle (SDL)
## -----------------------------------------------------------------------------

if(UNIX)
  append(OIDN_C_CXX_FLAGS "-fPIC -Wformat -Wformat-security")
  append(OIDN_C_CXX_FLAGS_RELEASE "-D_FORTIFY_SOURCE=2")
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
      append(OIDN_C_CXX_FLAGS "-fstack-protector-all")
    else()
      append(OIDN_C_CXX_FLAGS "-fstack-protector-strong")
    endif()

    # GCC might be very paranoid for partial structure initialization, e.g.
    #   struct { int a, b; } s = { 0, };
    # However the behavior is triggered by `Wmissing-field-initializers`
    # only. To prevent warnings on users' side who use the library and turn
    # this warning on, let's use it too. Applicable for the library sources
    # and interfaces only (tests currently rely on that fact heavily)
    append(OIDN_C_CXX_FLAGS "-Wmissing-field-initializers")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    append(OIDN_C_CXX_FLAGS "-fstack-protector-all")
  elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    append(CMAKE_CXX_FLAGS "-fstack-protector")
  endif()
  if(APPLE)
    append(CMAKE_SHARED_LINKER_FLAGS "-Wl,-bind_at_load")
    append(CMAKE_EXE_LINKER_FLAGS "-Wl,-bind_at_load")
  else()
    append(CMAKE_EXE_LINKER_FLAGS "-pie")
    append(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
    append(CMAKE_EXE_LINKER_FLAGS "-Wl,-z,noexecstack -Wl,-z,relro -Wl,-z,now")
  endif()
elseif(MSVC AND CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  append(OIDN_C_CXX_FLAGS "/guard:cf")
endif()

## -----------------------------------------------------------------------------
## Clang Sanitizer
## -----------------------------------------------------------------------------

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(OIDN_SANITIZER "None" CACHE STRING "Enables a Clang sanitizer.")
  set_property(CACHE OIDN_SANITIZER PROPERTY STRINGS "None" "Address" "Memory" "MemoryWithOrigin" "Undefined")
  mark_as_advanced(OIDN_SANITIZER)

  if(NOT OIDN_SANITIZER STREQUAL "None")
    if(OIDN_SANITIZER MATCHES "Memory(WithOrigin)?")
      append(OIDN_C_CXX_FLAGS "-fsanitize=memory")
      if(OIDN_SANITIZER STREQUAL "MemoryWithOrigin")
        append(OIDN_C_CXX_FLAGS "-fsanitize-memory-track-origins=2")
        append(OIDN_C_CXX_FLAGS "-fno-omit-frame-pointer")
      endif()
    elseif(OIDN_SANITIZER STREQUAL "Undefined")
      append(OIDN_C_CXX_FLAGS "-fsanitize=undefined")
      append(OIDN_C_CXX_FLAGS "-fno-sanitize=function,vptr") # work around linking problems
      append(OIDN_C_CXX_FLAGS "-fno-omit-frame-pointer")
    elseif(OIDN_SANITIZER STREQUAL "Address")
      append(OIDN_C_CXX_FLAGS "-fsanitize=address")
    elseif(OIDN_SANITIZER STREQUAL "Thread")
      append(OIDN_C_CXX_FLAGS "-fsanitize=thread")
    elseif(OIDN_SANITIZER STREQUAL "Leak")
      append(OIDN_C_CXX_FLAGS "-fsanitize=leak")
    else()
      message(FATAL_ERROR "Unsupported Clang sanitizer '${OIDN_SANITIZER}'")
    endif()
    message(STATUS "Using Clang ${OIDN_SANITIZER} sanitizer (experimental!)")
    append(OIDN_C_CXX_FLAGS "-g -fno-omit-frame-pointer")
  endif()
endif()

## -----------------------------------------------------------------------------
## Set flags
## -----------------------------------------------------------------------------

append(CMAKE_C_FLAGS   "${OIDN_C_CXX_FLAGS}")
append(CMAKE_CXX_FLAGS "${OIDN_C_CXX_FLAGS} ${OIDN_CXX_FLAGS}")

append(CMAKE_C_FLAGS_RELEASE   "${OIDN_C_CXX_FLAGS_RELEASE}")
append(CMAKE_CXX_FLAGS_RELEASE "${OIDN_C_CXX_FLAGS_RELEASE}")

append(CMAKE_C_FLAGS_RELWITHDEBINFO   "${OIDN_C_CXX_FLAGS_RELEASE}")
append(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${OIDN_C_CXX_FLAGS_RELEASE}")

append(CMAKE_C_FLAGS_DEBUG   "${OIDN_C_CXX_FLAGS_DEBUG}")
append(CMAKE_CXX_FLAGS_DEBUG "${OIDN_C_CXX_FLAGS_DEBUG}")
