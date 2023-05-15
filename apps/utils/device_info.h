// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/common.h"

OIDN_NAMESPACE_BEGIN

  inline int printPhysicalDevices()
  {
    const int numDevices = getNumPhysicalDevices();
    if (numDevices == 0)
    {
      std::cout << "No supported devices found" << std::endl;
      return 1;
    }

    for (int i = 0; i < numDevices; ++i)
    {
      PhysicalDeviceRef physicalDevice(i);
      std::cout << "Device " << i << std::endl;
      std::cout << "  Name: " << physicalDevice.get<std::string>("name") << std::endl;
      std::cout << "  Type: " << physicalDevice.get<DeviceType>("type") << std::endl;
      if (physicalDevice.get<bool>("uuidSupported"))
        std::cout << "  UUID: " << physicalDevice.get<OIDN_NAMESPACE::UUID>("uuid") << std::endl;
      if (physicalDevice.get<bool>("luidSupported"))
      {
        std::cout << "  LUID: " << physicalDevice.get<OIDN_NAMESPACE::LUID>("luid") << std::endl;
        std::cout << "  Node: " << physicalDevice.get<uint32_t>("nodeMask") << std::endl;
      }
      if (physicalDevice.get<bool>("pciAddressSupported"))
      {
        auto flags = std::cout.flags();
        std::cout << "  PCI : "
                  << std::hex << std::setfill('0')
                  << std::setw(4) << physicalDevice.get<int>("pciDomain") << ":"
                  << std::setw(2) << physicalDevice.get<int>("pciBus")    << ":"
                  << std::setw(2) << physicalDevice.get<int>("pciDevice") << "."
                  << std::setw(1) << physicalDevice.get<int>("pciFunction")
                  << std::endl;
        std::cout.flags(flags);
      }
      if (i < numDevices-1)
        std::cout << std::endl;
    }

    return 0;
  }

OIDN_NAMESPACE_END
