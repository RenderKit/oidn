// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_device.h"
#include "platform_ispc.h"

#if defined(OIDN_DNNL)
  #include "dnnl/dnnl_engine.h"
#elif defined(OIDN_BNNS)
  #include "bnns/bnns_engine.h"
#else
  #include "cpu_engine.h"
#endif

#if defined(OIDN_ARCH_X64)
  #if defined(_WIN32)
    #include <intrin.h> // __cpuid
  #elif !defined(__APPLE__)
    #include <cpuid.h>
  #endif
#endif

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_ARCH_X64) && !defined(__APPLE__)
  oidn_inline void cpuid(int cpuInfo[4], int functionID)
  {
  #if defined(_WIN32)
    __cpuid(cpuInfo, functionID);
  #else
    __cpuid(functionID, cpuInfo[0], cpuInfo[1], cpuInfo[2], cpuInfo[3]);
  #endif
  }
#endif

  CPUPhysicalDevice::CPUPhysicalDevice(int score)
    : PhysicalDevice(DeviceType::CPU, score)
  {
    name = CPUDevice::getName();
  }

  std::vector<Ref<PhysicalDevice>> CPUDevice::getPhysicalDevices()
  {
    CPUArch arch = getArch();
    if (arch == CPUArch::Unknown)
      return {};

    // Prefer the CPU over some low-power integrated GPUs
    int score = (1 << 16) + 61;
    return {makeRef<CPUPhysicalDevice>(score)};
  }

  std::string CPUDevice::getName()
  {
  #if defined(__APPLE__)
    char name[256] = {};
    size_t nameSize = sizeof(name)-1;
    if (sysctlbyname("machdep.cpu.brand_string", &name, &nameSize, nullptr, 0) == 0 && strlen(name) > 0)
      return name;
  #elif defined(OIDN_ARCH_X64)
    int regs[3][4];
    char name[sizeof(regs)+1] = {};

    cpuid(regs[0], 0x80000000);
    if (static_cast<unsigned int>(regs[0][0]) >= 0x80000004)
    {
      cpuid(regs[0], 0x80000002);
      cpuid(regs[1], 0x80000003);
      cpuid(regs[2], 0x80000004);
      memcpy(name, regs, sizeof(regs));
      if (strlen(name) > 0)
        return name;
    }
  #endif

    return "CPU"; // fallback
  }

  CPUArch CPUDevice::getArch()
  {
    switch (ispc::getCPUArch())
    {
    case ispc::CPUArch_SSE4:   return CPUArch::SSE41;
    case ispc::CPUArch_AVX2:   return CPUArch::AVX2;
    case ispc::CPUArch_AVX512: return CPUArch::AVX512;
    case ispc::CPUArch_NEON:   return CPUArch::NEON;
    default:                   return CPUArch::Unknown;
    }
  }

  CPUDevice::CPUDevice()
  {
    systemMemorySupported  = true;
    managedMemorySupported = true;

    // Get default values from environment variables
    getEnvVar("OIDN_NUM_THREADS", numThreads);
    getEnvVar("OIDN_SET_AFFINITY", setAffinity);
  }

  void CPUDevice::init()
  {
    arch = getArch();

    tensorDataType = DataType::Float32;
    weightDataType = DataType::Float32;

    std::unique_ptr<CPUEngine> engine;

  #if defined(OIDN_DNNL)
    if (arch == CPUArch::AVX512)
    {
      tensorLayout = TensorLayout::Chw16c;
      weightLayout = TensorLayout::OIhw16i16o;
      tensorBlockC = 16;
    }
    else
    {
      tensorLayout = TensorLayout::Chw8c;
      weightLayout = TensorLayout::OIhw8i8o;
      tensorBlockC = 8;
    }

    engine.reset(new DNNLEngine(this, numThreads));
  #elif defined(OIDN_BNNS)
    tensorLayout = TensorLayout::chw;
    weightLayout = TensorLayout::oihw;
    tensorBlockC = 1;

    engine.reset(new BNNSEngine(this, numThreads));
  #else
    if (arch == CPUArch::AVX512)
    {
      tensorLayout = TensorLayout::Chw16c;
      weightLayout = TensorLayout::IOhw16i16o;
      tensorBlockC = 16;
    }
    else
    {
      tensorLayout = TensorLayout::Chw8c;
      weightLayout = TensorLayout::IOhw8i8o;
      tensorBlockC = 8;
    }

    engine.reset(new CPUEngine(this, numThreads));
  #endif

    numThreads = engine->arena->max_concurrency();
    setAffinity = bool(engine->affinity);

    subdevices.emplace_back(new Subdevice(std::move(engine)));

    if (isVerbose())
    {
      std::cout << "  Device    : " << getName() << std::endl;
      std::cout << "    Type    : CPU" << std::endl;
      std::cout << "    ISA     : ";
      switch (arch)
      {
      case CPUArch::SSE2:   std::cout << "SSE2";    break;
      case CPUArch::SSE41:  std::cout << "SSE4.1";  break;
      case CPUArch::AVX2:   std::cout << "AVX2";    break;
      case CPUArch::AVX512: std::cout << "AVX-512"; break;
      case CPUArch::NEON:   std::cout << "NEON";    break;
      default:              std::cout << "Unknown"; break;
      }
      std::cout << std::endl;

      std::cout << "  Tasking   :";
      std::cout << " TBB" << TBB_VERSION_MAJOR << "." << TBB_VERSION_MINOR;
    #if TBB_INTERFACE_VERSION >= 12002
      std::cout << " TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << TBB_runtime_interface_version();
    #else
      std::cout << " TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << tbb::TBB_runtime_interface_version();
    #endif
      std::cout << std::endl;
      std::cout << "    Threads : " << numThreads << " (" << (setAffinity ? "affinitized" : "non-affinitized") << ")" << std::endl;
    }
  }

  Storage CPUDevice::getPtrStorage(const void* ptr)
  {
    return Storage::Host;
  }

  int CPUDevice::getInt(const std::string& name)
  {
    if (name == "numThreads")
      return numThreads;
    else if (name == "setAffinity")
      return setAffinity;
    else
      return Device::getInt(name);
  }

  void CPUDevice::setInt(const std::string& name, int value)
  {
    if (name == "numThreads")
    {
      if (!isEnvVar("OIDN_NUM_THREADS"))
        numThreads = value;
      else if (numThreads != value)
        printWarning("OIDN_NUM_THREADS environment variable overrides device parameter");
    }
    else if (name == "setAffinity")
    {
      if (!isEnvVar("OIDN_SET_AFFINITY"))
        setAffinity = value;
      else if (setAffinity != bool(value))
        printWarning("OIDN_SET_AFFINITY environment variable overrides device parameter");
    }
    else
      Device::setInt(name, value);

    dirty = true;
  }

  void CPUDevice::wait()
  {
    for (auto& subdevice : subdevices)
      subdevice->getEngine()->wait();
  }

OIDN_NAMESPACE_END
