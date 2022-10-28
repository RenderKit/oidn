// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_device.h"
#include "sycl_engine.h"
#include <iomanip>

namespace oidn {

  class SYCLDeviceSelector
  {
  public:
    int operator()(const sycl::device& device) const
    {
      if (!SYCLDevice::isDeviceSupported(device))
        return -1;

      // FIXME: improve detection of fastest discrete GPU
      return device.get_info<sycl::info::device::max_compute_units>() * device.get_info<sycl::info::device::max_work_group_size>();
    }
  };

  bool SYCLDevice::isSupported()
  {
    for (const auto& platform : sycl::platform::get_platforms())
      for (const auto& device : platform.get_devices(sycl::info::device_type::gpu))
        if (isDeviceSupported(device))
          return true;
    return false;
  }

  bool SYCLDevice::isDeviceSupported(const sycl::device& device)
  {
    return device.is_gpu() &&
           device.get_info<sycl::info::device::vendor_id>() == 0x8086 && // Intel
           device.has(sycl::aspect::usm_host_allocations) &&
           device.has(sycl::aspect::usm_device_allocations) &&
           device.has(sycl::aspect::usm_shared_allocations);
  }

  SYCLArch SYCLDevice::getDeviceArch(const sycl::device& device)
  {
    // FIXME: improve robustness
    if (device.get_info<sycl::info::device::max_work_group_size>() >= 1024)
    {
      if (device.has(sycl::aspect::fp64))
        return SYCLArch::XeHPC;
      else
        return SYCLArch::XeHPG;
    }
    else
      return SYCLArch::Gen9;
  }

  SYCLDevice::SYCLDevice(const std::vector<sycl::queue>& queues)
    : queues(queues)
  {
    // Get default values from environment variables
    getEnvVar("OIDN_NUM_SUBDEVICES", numSubdevices);
  }

  void SYCLDevice::init()
  {
    if (queues.empty())
    {
      try
      {
        sycl::device device {SYCLDeviceSelector()};

        arch = getDeviceArch(device);

        // Try to split the device into sub-devices per NUMA domain (tile)
        const auto partition = sycl::info::partition_property::partition_by_affinity_domain;
        const auto supportedPartitions = device.get_info<sycl::info::device::partition_properties>();
        if (std::find(supportedPartitions.begin(), supportedPartitions.end(), partition) != supportedPartitions.end())
        {
          const auto domain = sycl::info::partition_affinity_domain::numa;
          const auto supportedDomains = device.get_info<sycl::info::device::partition_affinity_domains>();
          if (std::find(supportedDomains.begin(), supportedDomains.end(), domain) != supportedDomains.end())
          {
            auto subDevices = device.create_sub_devices<partition>(domain);
            for (auto& subDevice : subDevices)
              queues.emplace_back(subDevice);
          }
        }

        if (queues.empty())
          queues.emplace_back(device);
      }
      catch (sycl::exception& e)
      {
        if (e.code() == sycl::errc::runtime)
          throw Exception(Error::UnsupportedHardware, "no supported SYCL device found");
        else
          throw;
      }
    }
    else
    {
      for (size_t i = 0; i < queues.size(); ++i)
      {
        if (!isDeviceSupported(queues[i].get_device()))
          throw Exception(Error::UnsupportedHardware, "unsupported SYCL device");

        if (i == 0)
        {
          arch = getDeviceArch(queues[i].get_device());
        }
        else
        {
          if (getDeviceArch(queues[i].get_device()) != arch)
            throw Exception(Error::UnsupportedHardware, "unsupported mixture of SYCL devices");
        }
      }
    }
    
    // Limit the number of subdevices/engines if requested
    if (numSubdevices > 0 && numSubdevices < int(queues.size()))
      queues.resize(numSubdevices);
    numSubdevices = int(queues.size());

    if (isVerbose())
    {
      std::cout << "  Platform  : " << queues[0].get_device().get_platform().get_info<sycl::info::platform::name>() << std::endl;
      
      for (size_t i = 0; i < queues.size(); ++i)
      { 
        if (queues.size() > 1)
           std::cout << "  Device " << std::setw(2) << i << " : ";
        else
          std::cout << "  Device    : ";
        std::cout << queues[i].get_device().get_info<sycl::info::device::name>() << std::endl;
        
        std::cout << "    Arch    : ";
        switch (arch)
        {
        case SYCLArch::Gen9:  std::cout << "Gen9/Gen11/Xe-LP"; break;
        case SYCLArch::XeHPG: std::cout << "Xe-HPG";     break;
        case SYCLArch::XeHPC: std::cout << "Xe-HPC";     break;
        default:              std::cout << "Unknown";
        }
        std::cout << std::endl;
        
        std::cout << "    EUs     : " << queues[i].get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
      }
    }

    // Create the engines
    for (auto& queue : queues)
      engines.push_back(makeRef<SYCLEngine>(this, queue));

    queues.clear(); // not needed anymore

    tensorDataType  = DataType::Float16;
    tensorLayout    = TensorLayout::Chw16c;
    tensorBlockSize = 16;

    switch (arch)
    {
    case SYCLArch::XeHPG:
      weightsLayout = TensorLayout::OIhw2o8i8o2i;
      break;
    case SYCLArch::XeHPC:
      weightsLayout = TensorLayout::OIhw8i16o2i;
      break;
    default:
      weightsLayout = TensorLayout::OIhw16i16o;
    }
  }
  
  int SYCLDevice::get1i(const std::string& name)
  {
    if (name == "numSubdevices")
      return numSubdevices;
    else
      return Device::get1i(name);
  }

  void SYCLDevice::set1i(const std::string& name, int value)
  {
    if (name == "numSubdevices")
    {
      if (!isEnvVar("OIDN_NUM_SUBDEVICES"))
        numSubdevices = value;
      else if (numSubdevices != value)
        warning("OIDN_NUM_SUBDEVICES environment variable overrides device parameter");
    }
    else
      Device::set1i(name, value);

    dirty = true;
  }
  
  void SYCLDevice::submitBarrier()
  {
    // We need a barrier only if there are at least 2 engines
    if (engines.size() < 2)
      return;

    // Submit the barrier to the default engine
    // The barrier depends on the commands on all engines
    engines[0]->depEvents = getDoneEvents();
    engines[0]->submitBarrier();
    
    // The next commands on all the other engines also depend on the barrier
    for (size_t i = 1; i < engines.size(); ++i)
    {
      engines[i]->lastEvent.reset();
      engines[i]->depEvents = {engines[0]->lastEvent.value()};
    }
  }

  void SYCLDevice::wait()
  {
    // Wait for the commands on all engines to complete
    sycl::event::wait_and_throw(getDoneEvents());
    
    // We can now discard all events
    for (auto& engine : engines)
      engine->lastEvent.reset();
  }
  
  void SYCLDevice::setDepEvents(const std::vector<sycl::event>& depEvents)
  {
    for (auto& engine : engines)
    {
      engine->lastEvent.reset();
      engine->depEvents = depEvents;
    }
  }
  
  std::vector<sycl::event> SYCLDevice::getDoneEvents()
  {
    std::vector<sycl::event> events;
    for (auto& engine : engines)
    {
      if (engine->lastEvent)
        events.push_back(engine->lastEvent.value());
    }
    return events;
  }

} // namespace oidn
