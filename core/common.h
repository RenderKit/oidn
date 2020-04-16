// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

#include "mkl-dnn/include/dnnl.hpp"
#include "mkl-dnn/include/dnnl_debug.h"
#include "mkl-dnn/src/common/dnnl_thread.hpp"
#include "mkl-dnn/src/common/type_helpers.hpp"
#include "mkl-dnn/src/cpu/jit_generator.hpp"

#include "common/ref.h"
#include "common/exception.h"
#include "common/thread.h"
#include "common/tasking.h"
#include "common/math.h"
#include "input_reorder_ispc.h" // ispc::Memory, ispc::Image

namespace oidn {

  using namespace dnnl;
  using namespace dnnl::impl::cpu;
  using dnnl::impl::parallel_nd;
  using dnnl::impl::memory_desc_matches_tag;

  inline size_t getFormatBytes(Format format)
  {
    switch (format)
    {
    case Format::Undefined: return 1;
    case Format::Float:     return sizeof(float);
    case Format::Float2:    return sizeof(float)*2;
    case Format::Float3:    return sizeof(float)*3;
    case Format::Float4:    return sizeof(float)*4;
    }
    assert(0);
    return 0;
  }

} // namespace oidn
