// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#include "common/ref.h"
#include "common/exception.h"
#include "common/thread.h"
#include "common/tasking.h"
#include "common/math.h"

#include "mkl-dnn/include/dnnl.hpp"
#include "mkl-dnn/include/dnnl_debug.h"
#include "mkl-dnn/src/common/dnnl_thread.hpp"
#include "mkl-dnn/src/cpu/x64/cpu_isa_traits.hpp"

#include "input_reorder_ispc.h" // ispc::Tensor, ispc::Image

namespace oidn {

  namespace x64 = dnnl::impl::cpu::x64;
  using dnnl::impl::parallel_nd;

  // Returns the size of the format in bytes
  __forceinline size_t getByteSize(Format format)
  {
    switch (format)
    {
    case Format::Undefined: return 1;
    case Format::Float:     return sizeof(float);
    case Format::Float2:    return sizeof(float)*2;
    case Format::Float3:    return sizeof(float)*3;
    case Format::Float4:    return sizeof(float)*4;
    default:                assert(0);
    }
    return 0;
  }

} // namespace oidn
