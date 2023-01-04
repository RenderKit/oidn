// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_device.h"
#if defined(OIDN_DNNL)
  #include "../dnnl/dnnl_engine.h"
#elif defined(OIDN_BNNS)
  #include "../bnns/bnns_engine.h"
#endif

#if defined(OIDN_ARCH_X64)
  #include "mkl-dnn/src/cpu/x64/xbyak/xbyak_util.h"
#endif

namespace oidn {

  bool CPUDevice::isSupported()
  {
    return getArch() != CPUArch::Unknown;
  }

  CPUArch CPUDevice::getArch()
  {
  #if defined(OIDN_ARCH_X64)
    using Xbyak::util::Cpu;
    static Cpu cpu;

    if (cpu.has(Cpu::tAVX512F)  && cpu.has(Cpu::tAVX512BW) &&
        cpu.has(Cpu::tAVX512VL) && cpu.has(Cpu::tAVX512DQ))
        return CPUArch::AVX512_CORE;

    if (cpu.has(Cpu::tAVX2))
      return CPUArch::AVX2;

    if (cpu.has(Cpu::tSSE41))
      return CPUArch::SSE41;

    return CPUArch::Unknown;
  #elif defined(OIDN_ARCH_ARM64)
    return CPUArch::NEON;
  #else
    return CPUArch::Unknown;
  #endif
  }

  CPUDevice::CPUDevice()
  {
    // Get default values from environment variables
    getEnvVar("OIDN_NUM_THREADS", numThreads);
    getEnvVar("OIDN_SET_AFFINITY", setAffinity);
  }

  CPUDevice::~CPUDevice()
  {
    observer.reset();
  }

  void CPUDevice::init()
  {
    arch = getArch();
    initTasking();

  #if defined(OIDN_DNNL)
    tensorDataType = DataType::Float32;
    if (arch == CPUArch::AVX512_CORE)
    {
      tensorLayout  = TensorLayout::Chw16c;
      weightsLayout = TensorLayout::OIhw16i16o;
      tensorBlockC  = 16;
    }
    else
    {
      tensorLayout  = TensorLayout::Chw8c;
      weightsLayout = TensorLayout::OIhw8i8o;
      tensorBlockC  = 8;
    }
  #else
    tensorLayout  = TensorLayout::chw;
    weightsLayout = TensorLayout::oihw;
    tensorBlockC  = 1;
  #endif

    if (isVerbose())
    {
      // FIXME: detect CPU name
      std::cout << "  Device    : CPU" << std::endl;
      std::cout << "    Arch    : ";
      switch (arch)
      {
      case CPUArch::AVX512_CORE:
        std::cout << "AVX512";
        break;
      case CPUArch::AVX2:
        std::cout << "AVX2";
        break;
      case CPUArch::SSE41:
        std::cout << "SSE4.1";
        break;
      case CPUArch::NEON:
        std::cout << "NEON";
        break;
      default:
        std::cout << "Unknown";
      }
      std::cout << std::endl;
    }

  #if defined(OIDN_DNNL)
    engine = makeRef<DNNLEngine>(this);
  #elif defined(OIDN_BNNS)
    engine = makeRef<BNNSEngine>(this);
  #endif
  }

  void CPUDevice::initTasking()
  {
    // Get the thread affinities for one thread per core on non-hybrid CPUs with SMT
  #if !(defined(__APPLE__) && defined(OIDN_ARCH_ARM64))
    if (setAffinity
      #if TBB_INTERFACE_VERSION >= 12020 // oneTBB 2021.2 or later
        && tbb::info::core_types().size() <= 1 // non-hybrid cores
      #endif
       )
    {
      affinity = std::make_shared<ThreadAffinity>(1, verbose);
      if (affinity->getNumThreads() == 0 ||                                           // detection failed
          tbb::this_task_arena::max_concurrency() == affinity->getNumThreads() ||     // no SMT
          (tbb::this_task_arena::max_concurrency() % affinity->getNumThreads()) != 0) // hybrid SMT
        affinity.reset(); // disable affinitization
    }
  #endif

    // Create the task arena
    const int maxNumThreads = affinity ? affinity->getNumThreads() : tbb::this_task_arena::max_concurrency();
    numThreads = (numThreads > 0) ? min(numThreads, maxNumThreads) : maxNumThreads;
    arena = std::make_shared<tbb::task_arena>(numThreads);

    // Automatically set the thread affinities
    if (affinity)
      observer = std::make_shared<PinningObserver>(affinity, *arena);

    if (isVerbose())
    {
      std::cout << "  Tasking   :";
      std::cout << " TBB" << TBB_VERSION_MAJOR << "." << TBB_VERSION_MINOR;
    #if TBB_INTERFACE_VERSION >= 12002
      std::cout << " TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << TBB_runtime_interface_version();
    #else
      std::cout << " TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << tbb::TBB_runtime_interface_version();
    #endif
      std::cout << std::endl;
      std::cout << "    Threads : " << numThreads << " (" << (affinity ? "affinitized" : "non-affinitized") << ")" << std::endl;
    }
  }

  Storage CPUDevice::getPointerStorage(const void* ptr)
  {
    return Storage::Host;
  }

  int CPUDevice::get1i(const std::string& name)
  {
    if (name == "numThreads")
      return numThreads;
    else if (name == "setAffinity")
      return setAffinity;
    else
      return Device::get1i(name);
  }

  void CPUDevice::set1i(const std::string& name, int value)
  {
    if (name == "numThreads")
    {
      if (!isEnvVar("OIDN_NUM_THREADS"))
        numThreads = value;
      else if (numThreads != value)
        warning("OIDN_NUM_THREADS environment variable overrides device parameter");
    }
    else if (name == "setAffinity")
    {
      if (!isEnvVar("OIDN_SET_AFFINITY"))
        setAffinity = value;
      else if (setAffinity != bool(value))
        warning("OIDN_SET_AFFINITY environment variable overrides device parameter");
    }
    else
      Device::set1i(name, value);

    dirty = true;
  }

  void CPUDevice::wait()
  {
    engine->wait();
  }

} // namespace oidn
