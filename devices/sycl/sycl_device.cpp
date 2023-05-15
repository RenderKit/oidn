// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_device.h"
#include "sycl_device_ids.h"
#include "sycl_engine.h"
#include <iomanip>

OIDN_NAMESPACE_BEGIN

  class SYCLDeviceSelector
  {
  public:
    int operator()(const sycl::device& syclDevice) const
    {
      const sycl::backend syclBackend = syclDevice.get_backend();
      const int score = SYCLDevice::getScore(syclDevice);

      if (syclBackend == sycl::backend::ext_oneapi_level_zero)
        return score * 2 + 1; // prefer Level Zero
      else if (syclBackend == sycl::backend::opencl)
        return score * 2;
      else
        return -1; // other backends are not supported
    }
  };

  SYCLPhysicalDevice::SYCLPhysicalDevice(const sycl::device& syclDevice, int score)
    : PhysicalDevice(DeviceType::SYCL, score),
      syclDevice(syclDevice)
  {
    name = syclDevice.get_info<sycl::info::device::name>();

    if (syclDevice.get_backend() != sycl::backend::ext_oneapi_level_zero)
      return; // only Level Zero supports further features

    // Check the supported Level Zero extensions
  #if defined(_WIN32)
    ze_driver_handle_t zeDriver =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclDevice.get_platform());

    uint32_t numExtensions = 0;
    std::vector<ze_driver_extension_properties_t> extensions;
    if (zeDriverGetExtensionProperties(zeDriver, &numExtensions, extensions.data()) != ZE_RESULT_SUCCESS)
      return;
    extensions.resize(numExtensions);
    if (zeDriverGetExtensionProperties(zeDriver, &numExtensions, extensions.data()) != ZE_RESULT_SUCCESS)
      return;

    for (const auto& extension : extensions)
    {
      if (strcmp(extension.name, ZE_DEVICE_LUID_EXT_NAME) == 0)
        luidSupported = true;
    }
  #endif

    // Get the device UUID and LUID
    ze_device_handle_t zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclDevice);

    ze_device_properties_t zeDeviceProps{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    ze_device_luid_ext_properties_t zeDeviceLUIDProps{ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES};
    if (luidSupported)
      zeDeviceProps.pNext = &zeDeviceLUIDProps;

    if (zeDeviceGetProperties(zeDevice, &zeDeviceProps) != ZE_RESULT_SUCCESS)
      return;

    static_assert(ZE_MAX_DEVICE_UUID_SIZE == OIDN_UUID_SIZE, "unexpected UUID size");
    memcpy(uuid.bytes, zeDeviceProps.uuid.id, sizeof(uuid.bytes));
    uuidSupported = true;

    if (luidSupported && zeDeviceLUIDProps.nodeMask != 0)
    {
      static_assert(ZE_MAX_DEVICE_LUID_SIZE_EXT == OIDN_LUID_SIZE, "unexpected LUID size");
      memcpy(luid.bytes, zeDeviceLUIDProps.luid.id, sizeof(luid.bytes));
      nodeMask = zeDeviceLUIDProps.nodeMask;
    }
    else
      luidSupported = false; // LUID may be invalid

    // Get the PCI address
    ze_pci_ext_properties_t zePCIProps{ZE_STRUCTURE_TYPE_PCI_EXT_PROPERTIES};
    if (zeDevicePciGetPropertiesExt(zeDevice, &zePCIProps) == ZE_RESULT_SUCCESS)
    {
      pciDomain   = zePCIProps.address.domain;
      pciBus      = zePCIProps.address.bus;
      pciDevice   = zePCIProps.address.device;
      pciFunction = zePCIProps.address.function;
      pciAddressSupported = true;
    }
  }

  std::vector<Ref<PhysicalDevice>> SYCLDevice::getPhysicalDevices()
  {
    std::vector<Ref<PhysicalDevice>> devices;

    // Select the backend to use to avoid duplicate devices
    const auto syclPlatforms = sycl::platform::get_platforms();
    sycl::backend syclBackend = sycl::backend::opencl; // fallback
    for (const auto& syclPlatform : syclPlatforms)
    {
      if (syclPlatform.get_backend() == sycl::backend::ext_oneapi_level_zero &&
          !syclPlatform.get_devices(sycl::info::device_type::gpu).empty())
      {
        syclBackend = sycl::backend::ext_oneapi_level_zero; // prefer Level Zero
        break;
      }
    }

    for (const auto& syclPlatform : syclPlatforms)
    {
      if (syclPlatform.get_backend() != syclBackend)
        continue;

      for (const auto& syclDevice : syclPlatform.get_devices(sycl::info::device_type::gpu))
      {
        const int score = getScore(syclDevice);
        if (score >= 0) // if supported
          devices.push_back(makeRef<SYCLPhysicalDevice>(syclDevice, score));
      }
    }

    return devices;
  }

  SYCLArch SYCLDevice::getArch(const sycl::device& syclDevice)
  {
    // Check whether the device supports the required features
    auto syclBackend = syclDevice.get_backend();
    if ((syclBackend != sycl::backend::ext_oneapi_level_zero && syclBackend != sycl::backend::opencl) ||
        !syclDevice.is_gpu() ||
        syclDevice.get_info<sycl::info::device::vendor_id>() != 0x8086 || // Intel
        !syclDevice.has(sycl::aspect::usm_host_allocations) ||
        !syclDevice.has(sycl::aspect::usm_device_allocations) ||
        !syclDevice.has(sycl::aspect::usm_shared_allocations) ||
        !syclDevice.has(sycl::aspect::ext_intel_device_id))
      return SYCLArch::Unknown;

    // Lookup the device ID to identify the architecture
    const unsigned int deviceID = syclDevice.get_info<sycl::ext::intel::info::device::device_id>();

    for (const auto& table : syclDeviceIDTables)
    {
      for (int tableID : table.ids)
      {
        if (tableID == deviceID)
          return table.arch;
      }
    }

    return SYCLArch::Unknown;
  }

  int SYCLDevice::getScore(const sycl::device& syclDevice)
  {
    const SYCLArch arch = getArch(syclDevice);
    if (arch == SYCLArch::Unknown)
      return -1;

    // Prefer the highest architecture GPU with the most compute units
    int score = 0;
    switch (arch)
    {
    case SYCLArch::XeLP:  score = 1;  break;
    case SYCLArch::XeHPG: score = 20; break;
    case SYCLArch::XeHPC: score = 30; break;
    default:
      return -1;
    }

    return (score << 16) + syclDevice.get_info<sycl::info::device::max_compute_units>();
  }

  SYCLDevice::SYCLDevice(const std::vector<sycl::queue>& syclQueues)
    : syclQueues(syclQueues)
  {
    preinit();
  }

  SYCLDevice::SYCLDevice(const Ref<SYCLPhysicalDevice>& physicalDevice)
    : physicalDevice(physicalDevice)
  {
    preinit();
  }

  void SYCLDevice::preinit()
  {
    managedMemorySupported = true;

    // Get default values from environment variables
    getEnvVar("OIDN_NUM_SUBDEVICES", numSubdevices);
  }

  void SYCLDevice::init()
  {
    if (syclQueues.empty())
    {
      // Create SYCL queues(s)
      try
      {
        sycl::device syclDevice = physicalDevice ? physicalDevice->syclDevice : sycl::device{SYCLDeviceSelector()};
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
          throw Exception(Error::UnsupportedHardware, "no supported SYCL devices found");
        else
          throw;
      }
    }
    else
    {
      // Check the specified SYCL queues
      for (size_t i = 0; i < syclQueues.size(); ++i)
      {
        const SYCLArch curArch = getArch(syclQueues[i].get_device());
        if (curArch == SYCLArch::Unknown)
          throw Exception(Error::UnsupportedHardware, "unsupported SYCL device");

        if (i == 0)
          arch = curArch;
        else
        {
          if (syclQueues[i].get_context() != syclQueues[0].get_context())
            throw Exception(Error::InvalidArgument, "SYCL queues belong to different SYCL contexts");
          if (curArch != arch)
            throw Exception(Error::InvalidArgument, "SYCL queues belong to devices with different architectures");
        }
      }
    }

    // Get the SYCL / Level Zero context
    syclContext = syclQueues[0].get_context();
    if (syclContext.get_platform().get_backend() == sycl::backend::ext_oneapi_level_zero)
      zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclContext);

    // Limit the number of subdevices/engines if requested
    if (numSubdevices > 0 && numSubdevices < int(syclQueues.size()))
      syclQueues.resize(numSubdevices);
    numSubdevices = int(syclQueues.size());

    // Print device info
    if (isVerbose())
    {
      std::cout << "  Backend   : " << syclContext.get_platform().get_info<sycl::info::platform::name>() << std::endl;

      for (size_t i = 0; i < syclQueues.size(); ++i)
      {
        if (syclQueues.size() > 1)
           std::cout << "  Subdev " << std::setw(2) << i << " : ";
        else
          std::cout << "  Device    : ";
        std::cout << syclQueues[i].get_device().get_info<sycl::info::device::name>() << std::endl;

        std::cout << "    Arch    : ";
        switch (arch)
        {
        case SYCLArch::XeLP:  std::cout << "Xe-LP";  break;
        case SYCLArch::XeHPG: std::cout << "Xe-HPG"; break;
        case SYCLArch::XeHPC: std::cout << "Xe-HPC"; break;
        default:              std::cout << "Unknown";
        }
        std::cout << std::endl;

        std::cout << "    EUs     : " << syclQueues[i].get_device().get_info<sycl::info::device::max_compute_units>() << std::endl;
      }
    }

    // Create the engines
    for (auto& syclQueue : syclQueues)
      engines.push_back(makeRef<SYCLEngine>(this, syclQueue));

    // Cleanup
    syclQueues.clear();
    physicalDevice.reset();

    // Set device properties
    tensorDataType = DataType::Float16;
    tensorLayout   = TensorLayout::Chw16c;
    tensorBlockC   = 16;

    switch (arch)
    {
    case SYCLArch::XeHPG:
      weightLayout = TensorLayout::OIhw2o8i8o2i;
      break;
    case SYCLArch::XeHPC:
      weightLayout = TensorLayout::OIhw8i16o2i;
      break;
    default:
      weightLayout = TensorLayout::OIhw16i16o;
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

  int SYCLDevice::getInt(const std::string& name)
  {
    if (name == "numSubdevices")
      return numSubdevices;
    else
      return Device::getInt(name);
  }

  void SYCLDevice::setInt(const std::string& name, int value)
  {
    if (name == "numSubdevices")
    {
      if (!isEnvVar("OIDN_NUM_SUBDEVICES"))
        numSubdevices = value;
      else if (numSubdevices != value)
        warning("OIDN_NUM_SUBDEVICES environment variable overrides device parameter");
    }
    else
      Device::setInt(name, value);

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
