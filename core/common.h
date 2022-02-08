// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#include "common/ref.h"
#include "common/exception.h"
#include "common/thread.h"
#include "common/tasking.h"
#include "common/math.h"
#include "vec.h"

#include "input_process_kernel_ispc.h" // ispc::TensorAccessor3D, ispc::ImageAccessor, ispc::Tile

namespace oidn {

  enum class DataType
  {
    Float32,
    Float16,
    UInt8,
  };

  template<typename T>
  struct DataTypeOf;

  template<> struct DataTypeOf<float>   { static constexpr DataType value = DataType::Float32; };
  template<> struct DataTypeOf<half>    { static constexpr DataType value = DataType::Float16; };
  template<> struct DataTypeOf<uint8_t> { static constexpr DataType value = DataType::UInt8;   };

  // Returns the size of a data type in bytes
  size_t getDataTypeSize(DataType dataType);

  // Returns the size of a format in bytes
  size_t getFormatSize(Format format);

  // Returns the data type of a format
  DataType getFormatDataType(Format format);

  std::ostream& operator <<(std::ostream& sm, Format format);

  template<typename T0, typename F>
  OIDN_INLINE void parallel_nd(const T0& D0, const F& f)
  {
    tbb::parallel_for(tbb::blocked_range<T0>(0, D0), [&](const tbb::blocked_range<T0>& r)
    {
      for (T0 i = r.begin(); i != r.end(); ++i)
        f(i);
    });
  }

  template<typename T0, typename T1, typename F>
  OIDN_INLINE void parallel_nd(const T0& D0, const T1& D1, const F& f)
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

} // namespace oidn
