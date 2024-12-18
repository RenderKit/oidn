// Copyright 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_device.h"
#include "sycl_device_table.h"
#include "sycl_engine.h"
#include <iomanip>

OIDN_NAMESPACE_BEGIN

  SYCLPhysicalDevice::SYCLPhysicalDevice(const sycl::device& syclDevice, int score)
    : PhysicalDevice(DeviceType::SYCL, score),
      syclDevice(syclDevice)
  {
    // Prevent the physical device from being automatically destroyed to avoid issues at process
    // exit. This is needed because the physical device is owned by the context which is static,
    // thus it might get destroyed *after* the SYCL runtime has been already unloaded (the module
    // unloading order is undefined). The resources held by the physical device will be released
    // at process exit anyway, so this intentional leak is fine.
    incRef();

    name = syclDevice.get_info<sycl::info::device::name>();

    if (syclDevice.get_backend() != sycl::backend::ext_oneapi_level_zero)
      return; // only Level Zero supports further features

    // Check the supported Level Zero extensions
    bool luidExtension = false;

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
      {
        luidExtension = true;
        break;
      }
    }
  #endif

    // Get the device UUID and LUID
    ze_device_handle_t zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclDevice);

    ze_device_properties_t zeDeviceProps{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    ze_device_luid_ext_properties_t zeDeviceLUIDProps{ZE_STRUCTURE_TYPE_DEVICE_LUID_EXT_PROPERTIES};
    if (luidExtension)
      zeDeviceProps.pNext = &zeDeviceLUIDProps;

    if (zeDeviceGetProperties(zeDevice, &zeDeviceProps) != ZE_RESULT_SUCCESS)
      return;

    static_assert(ZE_MAX_DEVICE_UUID_SIZE == OIDN_UUID_SIZE, "unexpected UUID size");
    memcpy(uuid.bytes, zeDeviceProps.uuid.id, sizeof(uuid.bytes));
    uuidSupported = true;

    if (luidExtension && zeDeviceLUIDProps.nodeMask != 0)
    {
      static_assert(ZE_MAX_DEVICE_LUID_SIZE_EXT == OIDN_LUID_SIZE, "unexpected LUID size");
      memcpy(luid.bytes, zeDeviceLUIDProps.luid.id, sizeof(luid.bytes));
      nodeMask = zeDeviceLUIDProps.nodeMask;
      luidSupported = true;
    }

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

    const auto syclPlatforms = sycl::platform::get_platforms();
    for (const auto& syclPlatform : syclPlatforms)
    {
      if (syclPlatform.get_backend() != sycl::backend::ext_oneapi_level_zero)
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

  bool SYCLDevice::isSupported(const sycl::device& syclDevice)
  {
    return getArch(syclDevice) != SYCLArch::Unknown;
  }

  SYCLArch SYCLDevice::getArch(const sycl::device& syclDevice)
  {
    // Check whether the device supports the required features
    if (syclDevice.get_backend() != sycl::backend::ext_oneapi_level_zero ||
        !syclDevice.is_gpu() ||
        syclDevice.get_info<sycl::info::device::vendor_id>() != 0x8086 || // Intel
        !syclDevice.has(sycl::aspect::usm_host_allocations) ||
        !syclDevice.has(sycl::aspect::usm_device_allocations) ||
        !syclDevice.has(sycl::aspect::usm_shared_allocations))
      return SYCLArch::Unknown;

    // Check the Level Zero driver version
    ze_driver_handle_t zeDriver =
      sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclDevice.get_platform());

    ze_driver_properties_t zeDriverProps{ZE_STRUCTURE_TYPE_DRIVER_PROPERTIES};
    if (zeDriverGetProperties(zeDriver, &zeDriverProps) != ZE_RESULT_SUCCESS)
      return SYCLArch::Unknown;

    if (zeDriverProps.driverVersion < 0x01036237) // 1.3.25143 (Windows Driver 31.0.101.4091)
      return SYCLArch::Unknown; // older versions do not work!

    // Check whether the device IP version is supported
    bool ipVersionSupported = false;
    uint32_t numExtensions = 0;
    std::vector<ze_driver_extension_properties_t> extensions;
    if (zeDriverGetExtensionProperties(zeDriver, &numExtensions, extensions.data()) != ZE_RESULT_SUCCESS)
      return SYCLArch::Unknown;
    extensions.resize(numExtensions);
    if (zeDriverGetExtensionProperties(zeDriver, &numExtensions, extensions.data()) != ZE_RESULT_SUCCESS)
      return SYCLArch::Unknown;

    for (const auto& extension : extensions)
    {
      if (strcmp(extension.name, ZE_DEVICE_IP_VERSION_EXT_NAME) == 0)
      {
        ipVersionSupported = true;
        break;
      }
    }

    if (!ipVersionSupported)
      return SYCLArch::Unknown;

    // Get the device IP version
    ze_device_handle_t zeDevice = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclDevice);
    ze_device_properties_t zeDeviceProps{ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES};
    ze_device_ip_version_ext_t zeDeviceIPVersion{ZE_STRUCTURE_TYPE_DEVICE_IP_VERSION_EXT};
    zeDeviceProps.pNext = &zeDeviceIPVersion;
    if (zeDeviceGetProperties(zeDevice, &zeDeviceProps) != ZE_RESULT_SUCCESS)
      return SYCLArch::Unknown;
    const uint32_t ipVersion = zeDeviceIPVersion.ipVersion & syclDeviceIPVersionMask; // remove revision

    // Lookup the IP version to identify the architecture
    for (const auto& entry : syclDeviceTable)
    {
      for (int entryIPVersion : entry.ipVersions)
      {
        if (entryIPVersion == ipVersion)
          return entry.arch;
      }
    }

  #if !defined(OIDN_DEVICE_SYCL_AOT)
    // Check whether ESIMD is supported
    // FIXME: enable when supported by ICX
    // if (!syclDevice.has(sycl::aspect::ext_intel_esimd))
    //   return SYCLArch::Unknown;

    // Get the EU SIMD width
    if (!syclDevice.has(sycl::aspect::ext_intel_gpu_eu_simd_width))
      return SYCLArch::Unknown;
    const int simdWidth = syclDevice.get_info<sycl::ext::intel::info::device::gpu_eu_simd_width>();

    // Gen 12.0.0 or newer is required
    if (ipVersion >= 0x03000000 && (simdWidth == 8 || simdWidth == 16))
      return SYCLArch::XeLP; // always fallback to Xe-LP
  #endif

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
    case SYCLArch::XeLP:         score = 1;  break;
    case SYCLArch::XeLPG:        score = 2;  break;
    case SYCLArch::XeLPGplus:    score = 10; break;
    case SYCLArch::XeHPG:        score = 20; break;
    case SYCLArch::XeHPC:        score = 30; break;
    case SYCLArch::XeHPC_NoDPAS: score = 20; break;
    case SYCLArch::Xe2LPG:       score = 11; break;
    case SYCLArch::Xe2HPG:       score = 21; break;
    case SYCLArch::Xe3LPG:       score = 12; break;
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
    if (syclQueues.empty() && physicalDevice)
    {
      // Create SYCL queues(s)
      try
      {
        sycl::device syclDevice = physicalDevice->syclDevice;
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
    else if (!syclQueues.empty())
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
    else
      throw Exception(Error::InvalidArgument, "no SYCL queues specified");

    // Get the SYCL / Level Zero context
    syclContext = syclQueues[0].get_context();
    if (syclContext.get_platform().get_backend() == sycl::backend::ext_oneapi_level_zero)
      zeContext = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(syclContext);

    // Limit the number of subdevices if requested
    if (numSubdevices > 0 && numSubdevices < int(syclQueues.size()))
      syclQueues.resize(numSubdevices);
    numSubdevices = int(syclQueues.size());

    // Print device info
    if (isVerbose())
    {
      for (size_t i = 0; i < syclQueues.size(); ++i)
      {
        sycl::device syclDevice = syclQueues[i].get_device();

        if (syclQueues.size() > 1)
           std::cout << "  Subdev " << std::setw(2) << i << " : ";
        else
          std::cout << "  Device    : ";
        std::cout << syclDevice.get_info<sycl::info::device::name>() << std::endl;

        std::cout << "    Type    : SYCL" << std::endl;
        std::cout << "    Arch    : ";
        switch (arch)
        {
        case SYCLArch::XeLP:         std::cout << "Xe-LP";   break;
        case SYCLArch::XeLPG:        std::cout << "Xe-LPG";  break;
        case SYCLArch::XeLPGplus:    std::cout << "Xe-LPG+"; break;
        case SYCLArch::XeHPG:        std::cout << "Xe-HPG";  break;
        case SYCLArch::XeHPC:        std::cout << "Xe-HPC";  break;
        case SYCLArch::XeHPC_NoDPAS: std::cout << "Xe-HPC";  break;
        case SYCLArch::Xe2LPG:       std::cout << "Xe2-LPG"; break;
        case SYCLArch::Xe2HPG:       std::cout << "Xe2-HPG"; break;
        case SYCLArch::Xe3LPG:       std::cout << "Xe3-LPG"; break;
        default:                     std::cout << "Unknown"; break;
        }
        std::cout << std::endl;

        if (syclDevice.has(sycl::aspect::ext_intel_gpu_eu_count))
          std::cout << "    EUs     : " << syclDevice.get_info<sycl::ext::intel::info::device::gpu_eu_count>() << std::endl;
      }

      std::cout << "  Backend   : " << syclContext.get_platform().get_info<sycl::info::platform::name>() << std::endl;
    }

    // Create the subdevices
    for (auto& syclQueue : syclQueues)
      subdevices.emplace_back(new Subdevice(std::unique_ptr<Engine>(new SYCLEngine(this, syclQueue))));

    // Cleanup
    syclQueues.clear();
    physicalDevice.reset();

    // Set device properties
    tensorDataType = DataType::Float16;
    tensorLayout   = TensorLayout::Chw16c;
    tensorBlockC   = 16;

    switch (arch)
    {
    case SYCLArch::XeLPGplus:
    case SYCLArch::XeHPG:
      weightDataType = DataType::Float16;
      weightLayout   = TensorLayout::OIhw2o8i8o2i;
      break;

    case SYCLArch::XeHPC:
    case SYCLArch::Xe2LPG:
    case SYCLArch::Xe2HPG:
    case SYCLArch::Xe3LPG:
      weightDataType = DataType::Float16;
      weightLayout   = TensorLayout::OIhw8i16o2i;
      break;

    default:
      weightDataType = DataType::Float32;
      weightLayout   = TensorLayout::OIhw16i16o;
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

  SYCLEngine* SYCLDevice::getSYCLEngine(int i) const
  {
    return static_cast<SYCLEngine*>(getEngine(i));
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
        printWarning("OIDN_NUM_SUBDEVICES environment variable overrides device parameter");
    }
    else
      Device::setInt(name, value);

    dirty = true;
  }

  Storage SYCLDevice::getPtrStorage(const void* ptr)
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
    // We need a barrier only if there are at least 2 subdevices
    const int numSubdevices = getNumSubdevices();
    if (numSubdevices < 2)
      return;

    // Submit the barrier to the main engine
    // The barrier depends on the commands on all engines
    SYCLEngine* mainEngine = getSYCLEngine(0);
    mainEngine->depEvents = getDoneEvents();
    mainEngine->submitBarrier();

    // The next commands on all the other engines also depend on the barrier
    for (int i = 1; i < numSubdevices; ++i)
    {
      SYCLEngine* engine = getSYCLEngine(i);
      engine->lastEvent.reset();
      engine->depEvents = {mainEngine->lastEvent.value()};
    }
  }

  void SYCLDevice::wait()
  {
    // Wait for the commands on all engines to complete
    sycl::event::wait_and_throw(getDoneEvents());

    // We can now discard all events
    for (int i = 0; i < getNumSubdevices(); ++i)
      getSYCLEngine(i)->lastEvent.reset();
  }

  void SYCLDevice::setDepEvents(const std::vector<sycl::event>& events)
  {
    for (int i = 0; i < getNumSubdevices(); ++i)
    {
      SYCLEngine* engine = getSYCLEngine(i);
      engine->lastEvent.reset();
      engine->depEvents = events;
    }
  }

  void SYCLDevice::setDepEvents(const sycl::event* events, int numEvents)
  {
    if (numEvents < 0)
      throw Exception(Error::InvalidArgument, "invalid number of dependent SYCL events");
    if (events == nullptr && numEvents > 0)
      throw Exception(Error::InvalidArgument, "array of dependent SYCL events is null");

    setDepEvents({events, events + numEvents});
  }

  std::vector<sycl::event> SYCLDevice::getDoneEvents()
  {
    std::vector<sycl::event> events;
    for (int i = 0; i < getNumSubdevices(); ++i)
    {
      SYCLEngine* engine = getSYCLEngine(i);
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
