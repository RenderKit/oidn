// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#include "common/ref.h"
#include "common/exception.h"
#include "common/thread.h"
#include "common/tasking.h"
#include "common/math.h"
#include "vec.h"

#if defined(OIDN_DNNL)
  #include "mkl-dnn/include/dnnl.hpp"
#elif defined(OIDN_BNNS)
  #include <Accelerate/Accelerate.h>
#endif

#include "input_reorder_ispc.h" // ispc::TensorAccessor, ispc::ImageAccessor, ispc::ReorderTile

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
  template<> struct DataTypeOf<uint8_t> { static constexpr DataType value = DataType::UInt8;   };

  // Returns the size of a data type in bytes
  __forceinline size_t getByteSize(DataType dataType)
  {
    switch (dataType)
    {
    case DataType::Float32: return 4;
    case DataType::Float16: return 2;
    case DataType::UInt8:   return 1;
    default:
      throw Exception(Error::Unknown, "invalid data type");
    }
  }

  // Returns the size of a format in bytes
  __forceinline size_t getByteSize(Format format)
  {
    switch (format)
    {
    case Format::Undefined: return 0;
    case Format::Float:     return sizeof(float);
    case Format::Float2:    return sizeof(float)*2;
    case Format::Float3:    return sizeof(float)*3;
    case Format::Float4:    return sizeof(float)*4;
    case Format::Half:      return sizeof(int16_t);
    case Format::Half2:     return sizeof(int16_t)*2;
    case Format::Half3:     return sizeof(int16_t)*3;
    case Format::Half4:     return sizeof(int16_t)*4;
    default:
      throw Exception(Error::Unknown, "invalid format");
    }
  }

  // Returns the data type of a format
  __forceinline DataType getDataType(Format format)
  {
    switch (format)
    {
    case Format::Float:
    case Format::Float2:
    case Format::Float3:
    case Format::Float4:
      return DataType::Float32;
    case Format::Half:
    case Format::Half2:
    case Format::Half3:
    case Format::Half4:
      return DataType::Float16;
    default:
      throw Exception(Error::Unknown, "invalid format");
    }
  }

  inline std::ostream& operator <<(std::ostream& sm, Format format)
  {
    switch (format)
    {
    case Format::Float:  sm << "f";  break;
    case Format::Float2: sm << "f2"; break;
    case Format::Float3: sm << "f3"; break;
    case Format::Float4: sm << "f4"; break;
    case Format::Half:   sm << "h";  break;
    case Format::Half2:  sm << "h2"; break;
    case Format::Half3:  sm << "h3"; break;
    case Format::Half4:  sm << "h4"; break;
    default:             sm << "?";  break;
    }
    return sm;
  }

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

} // namespace oidn
