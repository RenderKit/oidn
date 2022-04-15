// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#include "common/ref.h"
#include "common/exception.h"
#include "common/thread.h"
#include "common/math.h"
#include "vec.h"

#include "cpu_input_process_ispc.h" // ispc::TensorAccessor3D, ispc::ImageAccessor, ispc::Tile

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

} // namespace oidn
