// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common.h"

OIDN_NAMESPACE_BEGIN

  size_t getDataTypeSize(DataType dataType)
  {
    switch (dataType)
    {
    case DataType::Float32: return sizeof(float);
    case DataType::Float16: return sizeof(int16_t);
    case DataType::UInt8:   return 1;
    default:
      throw std::invalid_argument("invalid data type");
    }
  }

  DataType getFormatDataType(Format format)
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
      throw std::invalid_argument("invalid format");
    }
  }

OIDN_NAMESPACE_END