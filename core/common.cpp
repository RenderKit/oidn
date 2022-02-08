// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "common.h"

namespace oidn {

  size_t getDataTypeSize(DataType dataType)
  {
    switch (dataType)
    {
    case DataType::Float32: return sizeof(float);
    case DataType::Float16: return sizeof(int16_t);
    case DataType::UInt8:   return 1;
    default:
      throw Exception(Error::Unknown, "invalid data type");
    }
  }

  size_t getFormatSize(Format format)
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
      throw Exception(Error::Unknown, "invalid format");
    }
  }

  std::ostream& operator <<(std::ostream& sm, Format format)
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

} // namespace oidn