// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#include "common/ref.h"
#include "common/exception.h"
#include "common/thread.h"
#include "common/tasking.h"
#include "common/math.h"

#if defined(OIDN_DNNL)
  #include "mkl-dnn/include/dnnl.hpp"
  #include "mkl-dnn/include/dnnl_debug.h"
  #include "mkl-dnn/src/common/dnnl_thread.hpp"
  #include "mkl-dnn/src/cpu/x64/cpu_isa_traits.hpp"
#elif defined(OIDN_BNNS)
  #include <Accelerate/Accelerate.h>
#endif

#include "input_reorder_ispc.h" // ispc::Tensor, ispc::Image

namespace oidn {

#if defined(OIDN_DNNL)
  namespace x64 = dnnl::impl::cpu::x64;
  using dnnl::impl::parallel_nd;
#else
  template <typename T0, typename F>
  __forceinline void parallel_nd(const T0& D0, F f)
  {
    tbb::parallel_for(tbb::blocked_range<T0>(0, D0), [&](const tbb::blocked_range<T0>& r)
    {
      for (T0 i = r.begin(); i != r.end(); ++i)
        f(i);
    });
  }

  template <typename T0, typename T1, typename F>
  __forceinline void parallel_nd(const T0& D0, const T1& D1, F f)
  {
    tbb::parallel_for(tbb::blocked_range2d<T0, T1>(0, D0, 0, D1), [&](const tbb::blocked_range2d<T0, T1>& r)
    {
      for (T0 i = r.rows().begin(); i != r.rows().end(); ++i)
      {
        for (T1 j = r.cols().begin(); j != r.cols().end(); ++j)
          f(i, j);
      }
    });
  }
#endif

  // Returns the size of the format in bytes
  __forceinline size_t getByteSize(Format format)
  {
    switch (format)
    {
    case Format::Undefined: return 0;
    case Format::Float:     return sizeof(float);
    case Format::Float2:    return sizeof(float)*2;
    case Format::Float3:    return sizeof(float)*3;
    case Format::Float4:    return sizeof(float)*4;
    default:
      throw Exception(Error::Unknown, "invalid format");
    }
  }

} // namespace oidn
