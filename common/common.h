// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "oidn_utils.h" // must be included before platform.h
#include "platform.h"

OIDN_NAMESPACE_BEGIN

  // Synchronization mode for operations
  enum class SyncMode
  {
    Sync, // synchronous
    Async // asynchronous
  };

  // Data types sorted by precision in ascending order
  enum class DataType
  {
    UInt8,
    Float16,
    Float32,
  };

  template<typename T>
  struct DataTypeOf;

  template<> struct DataTypeOf<float>   { static constexpr DataType value = DataType::Float32; };
  template<> struct DataTypeOf<half>    { static constexpr DataType value = DataType::Float16; };
  template<> struct DataTypeOf<uint8_t> { static constexpr DataType value = DataType::UInt8;   };

  // Returns the size of a data type in bytes
  size_t getDataTypeSize(DataType dataType);

  // Returns the data type of a format
  DataType getFormatDataType(Format format);

OIDN_NAMESPACE_END