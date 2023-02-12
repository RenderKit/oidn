// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_device.h"
#include "sycl_engine.h"
#include <iomanip>

OIDN_NAMESPACE_BEGIN

  class SYCLDeviceSelector
  {
  public:
    int operator()(const sycl::device& syclDevice) const
    {
      const SYCLArch arch = SYCLDevice::getArch(syclDevice);
      if (arch == SYCLArch::Unknown)
        return -1;
    
      // Prefer the highest architecture device with the most compute units
      return (int(arch) << 24) | syclDevice.get_info<sycl::info::device::max_compute_units>();
    }
  };

  bool SYCLDevice::isSupported()
  {
    for (const auto& syclPlatform : sycl::platform::get_platforms())
      for (const auto& syclDevice : syclPlatform.get_devices(sycl::info::device_type::gpu))
        if (getArch(syclDevice) != SYCLArch::Unknown)
          return true;
    return false;
  }

  SYCLArch SYCLDevice::getArch(const sycl::device& syclDevice)
  {
    if (!syclDevice.is_gpu() ||
        syclDevice.get_info<sycl::info::device::vendor_id>() != 0x8086 || // Intel
        !syclDevice.has(sycl::aspect::usm_host_allocations) ||
        !syclDevice.has(sycl::aspect::usm_device_allocations) ||
        !syclDevice.has(sycl::aspect::usm_shared_allocations) ||
        !syclDevice.has(sycl::aspect::ext_intel_device_id))
      return SYCLArch::Unknown;

    const unsigned int deviceId = syclDevice.get_info<sycl::ext::intel::info::device::device_id>();
    const unsigned int maskedId = deviceId & 0xFF00;

    if (maskedId == 0x1600)
      return SYCLArch::Unknown; // unsupported: Gen8
    else if (maskedId == 0x4F00 || maskedId == 0x5600)
      return SYCLArch::XeHPG;
    else if (maskedId == 0x0B00)
      return SYCLArch::XeHPC;
    else
      return SYCLArch::Gen9; // fallback: Gen9, Gen10, Gen11, Xe-LP, Xe-HP, ...
  }

  SYCLDevice::SYCLDevice(const std::vector<sycl::queue>& syclQueues)
    : syclQueues(syclQueues)
  {
    // Get default values from environment variables
    getEnvVar("OIDN_NUM_SUBDEVICES", numSubdevices);
  }

  void SYCLDevice::init()
  {
    if (syclQueues.empty())
    {
      try
      {
        sycl::device syclDevice {SYCLDeviceSelector()};

        arch = getArch(syclDevice);

        // Try to split the device into sub-devices per NUMA domain (tile)
        const auto partition = sycl::info::partition_property::partition_by_affinity_domain;
        const auto supportedPartitions = syclDevice.get_info<sycl::info::device::partition_properties>();
        if (std::find(supportedPartitions.begin(), supportedPartitions.end(), partition) != supportedPartitions.end())
        {
          const auto domain = sycl::info::partition_affinity_domain::numa;
          const auto supportedDomains = syclDevice.get_info<sycl::info::device::partition_affinity_domains>();
          if (std::find(supportedDomains.begin(), supportedDomains.end(), domain) != supportedDomains.end())
          {
            auto syclSubDevices = syclDevice.create_sub_devices<partition>(domain);
            for (auto& syclSubDevice : syclSubDevices)
              syclQueues.emplace_back(syclSubDevice);
          }
        }

        if (syclQueues.empty())
          syclQueues.emplace_back(syclDevice);
      }
      catch (const sycl::exception& e)
      {
        if (e.code() == sycl::errc::runtime)
          throw Exception(Error::UnsupportedHardware, "no supported SYCL device found");
        else
          throw;
      }
    }
    else
    {
      for (size_t i = 0; i < syclQueues.size(); ++i)
      {
        const SYCLArch curArch = getArch(syclQueues[i].get_device());
        if (curArch == SYCLArch::Unknown)
          throw Exception(Error::UnsupportedHardware, "unsupported SYCL device");

        if (i == 0)
        {
          arch = curArch;
        }
        else
        {
          if (syclQueues[i].get_context() != syclQueues[0].get_context())
            throw Exception(Error::InvalidArgument, "queues belong to different SYCL contexts");
          if (curArch != arch)
            throw Exception(Error::UnsupportedHardware, "unsupported mixture of SYCL devices");
        }
      }
    }

    syclContext = syclQueues[0].get_context();

    if (syclContext.get_platform().get_backend() == sycl::backend::ext_oneapi_level_zero)
      zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclContext);
    
    // Limit the number of subdevices/engines if requested
    if (numSubdevices > 0 && numSubdevices < int(syclQueues.size()))
      syclQueues.resize(numSubdevices);
    numSubdevices = int(syclQueues.size());

    if (isVerbose())
    {
      std::cout << "  Platform  : " << syclContext.get_platform().get_info<sycl::info::platform::name>() << std::endl;
      
      for (size_t i = 0; i < syclQueues.size(); ++i)
      { 
        if (syclQueues.size() > 1)
           std::cout << "  Device " << std::setw(2) << i << " : ";
        else
          std::cout << "  Device    : ";
        std::cout << syclQueues[i].get_device().get_info<sycl::info::device::name>() << std::endl;
        
        std::cout << "    Arch    : ";
        switch (arch)
        {
        case SYCLArch::Gen9:  std::cout << "Gen9/Gen10/Gen11/Xe-LP/Xe-HP"; break;
        case SYCLArch::XeHPG: std::cout << "Xe-HPG";     break;
        case SYCLArch::XeHPC: std::cout << "Xe-HPC";     break;
        default:              std::cout << "Unknown";
        }
        std::cout << std::endl;
        
        std::cout << "    EUs     : " << syclQueues[i].get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
      }
    }

    // Create the engines
    for (auto& syclQueue : syclQueues)
      engines.push_back(makeRef<SYCLEngine>(this, syclQueue));

    syclQueues.clear(); // not needed anymore

    tensorDataType = DataType::Float16;
    tensorLayout   = TensorLayout::Chw16c;
    tensorBlockC   = 16;

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

    if (zeContext)
    {
    #if defined(_WIN32)
      externalMemoryTypes = ExternalMemoryTypeFlag::OpaqueWin32;
    #else
      externalMemoryTypes = ExternalMemoryTypeFlag::DMABuf;
    #endif
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

  Storage SYCLDevice::getPointerStorage(const void* ptr)
  {
    switch (sycl::get_pointer_type(ptr, syclContext))
    {
      case sycl::usm::alloc::host:
        return Storage::Host;
      case sycl::usm::alloc::device:
        return Storage::Device;
      case sycl::usm::alloc::shared:
        return Storage::Managed;
      default:
        return Storage::Undefined;
    }
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
  
  void SYCLDevice::setDepEvents(const std::vector<sycl::event>& events)
  {
    for (auto& engine : engines)
    {
      engine->lastEvent.reset();
      engine->depEvents = events;
    }
  }

  void SYCLDevice::setDepEvents(const sycl::event* events, int numEvents)
  {
    setDepEvents({events, events + numEvents});
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

  void SYCLDevice::getDoneEvent(sycl::event& event)
  {
    auto doneEvents = getDoneEvents();
    if (doneEvents.size() == 1)
      event = doneEvents[0];
    else if (doneEvents.size() == 0)
      event = {}; // no kernels were executed
    else
      throw std::logic_error("missing barrier after kernels");
  }

OIDN_NAMESPACE_END
