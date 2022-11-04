// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"
#include "common/ref.h"
#include "common/exception.h"
#include "common/thread.h"
#include "common/math.h"
#include "vec.h"

namespace oidn {

  // Synchronization mode for operations
  enum class SyncMode
  {
    Sync,  // synchronous
    Async  // asynchronous
  };

  // ---------------------------------------------------------------------------
  // Data types and formats
  // ---------------------------------------------------------------------------

  using math::vec3;
  using math::vec3f;

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
