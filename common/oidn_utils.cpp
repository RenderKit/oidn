// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "oidn_utils.h"

namespace oidn {

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
      throw std::invalid_argument("invalid format");
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

  std::ostream& operator <<(std::ostream& sm, DeviceType deviceType)
  {
    switch (deviceType)
    {
    case DeviceType::Default: sm << "default"; break;
    case DeviceType::CPU:     sm << "cpu";     break;
    case DeviceType::SYCL:    sm << "sycl";    break;
    case DeviceType::CUDA:    sm << "cuda";    break;
    case DeviceType::HIP:     sm << "hip";     break;
    default:
      throw std::invalid_argument("invalid device type");
    }
    
    return sm;
  }

  std::istream& operator >>(std::istream& sm, DeviceType& deviceType)
  {
    std::string str;
    sm >> str;

    if (str == "default" || str == "Default")
      deviceType = DeviceType::Default;
    else if (str == "cpu" || str == "CPU")
      deviceType = DeviceType::CPU;
    else if (str == "sycl" || str == "SYCL")
      deviceType = DeviceType::SYCL;
    else if (str == "cuda" || str == "CUDA")
      deviceType = DeviceType::CUDA;
    else if (str == "hip" || str == "HIP")
      deviceType = DeviceType::HIP;
    else
      throw std::invalid_argument("invalid device type");

    return sm;
  }

} // namespace oidn