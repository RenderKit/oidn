// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common.h"

OIDN_NAMESPACE_BEGIN

  size_t getDataTypeSize(DataType dataType)
  {
    switch (dataType)
    {
    case DataType::UInt8:   return 1;
    case DataType::Float16: return sizeof(int16_t);
    case DataType::Float32: return sizeof(float);
    default:
      throw std::invalid_argument("invalid data type");
    }
  }

  DataType getFormatDataType(Format format)
  {
    switch (format)
    {
    case Format::Undefined:
      return DataType::Void;
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

  Format makeFormat(DataType dataType, int numChannels)
  {
    if (dataType == DataType::Void)
      return Format::Undefined;

    Format baseFormat;
    switch (dataType)
    {
    case DataType::Float16:
      baseFormat = Format::Half;
      break;
    case DataType::Float32:
      baseFormat = Format::Float;
      break;
    default:
      throw std::invalid_argument("unsupported format data type");
    }

    if (numChannels < 1 || numChannels > 4)
      throw std::invalid_argument("invalid number of channels");

    return Format(int(baseFormat) + numChannels - 1);
  }

OIDN_NAMESPACE_END