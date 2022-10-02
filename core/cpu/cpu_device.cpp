// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_device.h"
#include "cpu_upsample.h"
#include "cpu_autoexposure.h"
#include "cpu_input_process.h"
#include "cpu_output_process.h"
#include "cpu_image_copy.h"

namespace oidn {

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
    // Initialize TBB
    initTasking();

    // Initialize the neural network runtime
  #if defined(OIDN_DNNL)
    dnnl_set_verbose(clamp(verbose - 2, 0, 2)); // unfortunately this is not per-device but global
    dnnlEngine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnlStream = dnnl::stream(dnnlEngine);
    tensorBlockSize = isISASupported(ISA::AVX512_CORE) ? 16 : 8;
  #else
    tensorBlockSize = 1;
  #endif
    tensorLayout   = tensorBlockSize == 16 ? TensorLayout::Chw16c : (tensorBlockSize == 8 ? TensorLayout::Chw8c : TensorLayout::chw);
    weightsLayout  = tensorBlockSize == 16 ? TensorLayout::OIhw16i16o : (tensorBlockSize == 8 ? TensorLayout::OIhw8i8o : TensorLayout::oihw);
    tensorDataType = DataType::Float32;

    if (isVerbose())
    {
      // FIXME: detect CPU name
      std::cout << "  Device    : CPU" << std::endl;
      std::cout << "    ISA     : ";
    #if defined(OIDN_ARCH_X64)
      if (isISASupported(ISA::AVX512_CORE))
        std::cout << "AVX512";
      else if (isISASupported(ISA::AVX2))
        std::cout << "AVX2";
      else if (isISASupported(ISA::SSE41))
        std::cout << "SSE4.1";
    #elif defined(OIDN_ARCH_ARM64)
      std::cout << "NEON";
    #endif
      std::cout << std::endl;
      
      std::cout << "  Neural    : ";
    #if defined(OIDN_DNNL)
      std::cout << "DNNL (oneDNN) " << DNNL_VERSION_MAJOR << "." <<
                                       DNNL_VERSION_MINOR << "." <<
                                       DNNL_VERSION_PATCH;
    #elif defined(OIDN_BNNS)
      std::cout << "BNNS";
    #endif
      std::cout << std::endl;
    }
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

  void CPUDevice::runHostTask(std::function<void()>&& f)
  {
    if (arena)
      arena->execute(f);
    else
      f();
  }

  std::shared_ptr<Upsample> CPUDevice::newUpsample(const UpsampleDesc& desc)
  {
    return std::make_shared<CPUUpsample>(this, desc);
  }

  std::shared_ptr<Autoexposure> CPUDevice::newAutoexposure(const ImageDesc& srcDesc)
  {
    return std::make_shared<CPUAutoexposure>(this, srcDesc);
  }

  std::shared_ptr<InputProcess> CPUDevice::newInputProcess(const InputProcessDesc& desc)
  {
    return std::make_shared<CPUInputProcess>(this, desc);
  }

  std::shared_ptr<OutputProcess> CPUDevice::newOutputProcess(const OutputProcessDesc& desc)
  {
    return std::make_shared<CPUOutputProcess>(this, desc);
  }

  std::shared_ptr<ImageCopy> CPUDevice::newImageCopy()
  {
    return std::make_shared<CPUImageCopy>(this);
  }

  void CPUDevice::runHostFuncAsync(std::function<void()>&& f)
  {
    f(); // no async execution on the CPU
  }

} // namespace oidn
