// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "oidn_utils.h"
#include "platform.h"

OIDN_NAMESPACE_BEGIN

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
    case DeviceType::CPU:     sm << "CPU";     break;
    case DeviceType::SYCL:    sm << "SYCL";    break;
    case DeviceType::CUDA:    sm << "CUDA";    break;
    case DeviceType::HIP:     sm << "HIP";     break;
    default:
      throw std::invalid_argument("invalid device type");
    }

    return sm;
  }

  std::istream& operator >>(std::istream& sm, DeviceType& deviceType)
  {
    std::string str;
    sm >> str;
    str = toLower(str);

    if (str == "default")
      deviceType = DeviceType::Default;
    else if (str == "cpu")
      deviceType = DeviceType::CPU;
    else if (str == "sycl")
      deviceType = DeviceType::SYCL;
    else if (str == "cuda")
      deviceType = DeviceType::CUDA;
    else if (str == "hip")
      deviceType = DeviceType::HIP;
    else
      throw std::invalid_argument("invalid device type");

    return sm;
  }

  std::ostream& operator <<(std::ostream& sm, Quality quality)
  {
    switch (quality)
    {
    case Quality::Default:  sm << "default";  break;
    case Quality::High:     sm << "high";     break;
    case Quality::Balanced: sm << "balanced"; break;
    default:
      throw std::invalid_argument("invalid quality mode");
    }
    return sm;
  }

  std::ostream& operator <<(std::ostream& sm, const UUID& uuid)
  {
    auto flags = sm.flags();
    for (size_t i = 0; i < sizeof(uuid.bytes); ++i)
      sm << std::hex << std::setw(2) << std::setfill('0') << int(uuid.bytes[i]);
    sm.flags(flags);
    return sm;
  }

  std::ostream& operator <<(std::ostream& sm, const LUID& luid)
  {
    auto flags = sm.flags();
    for (size_t i = 0; i < sizeof(luid.bytes); ++i)
      sm << std::hex << std::setw(2) << std::setfill('0') << int(luid.bytes[i]);
    sm.flags(flags);
    return sm;
  }

OIDN_NAMESPACE_END