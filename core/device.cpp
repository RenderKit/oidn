// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "device.h"
#include "scratch.h"
#include "unet.h"

namespace oidn {

  thread_local Device::ErrorState Device::globalError;

  Device::Device()
  {
  #if defined(OIDN_X64)
    if (!x64::mayiuse(x64::sse41))
      throw Exception(Error::UnsupportedHardware, "SSE4.1 support is required at minimum");
  #endif

    // Get default values from environment variables
    if (getEnvVar("OIDN_VERBOSE", verbose))
      error.verbose = verbose;
    getEnvVar("OIDN_NUM_THREADS", numThreads);
    getEnvVar("OIDN_SET_AFFINITY", setAffinity);
  }

  Device::~Device()
  {
    observer.reset();
  }

  void Device::setError(Device* device, Error code, const std::string& message)
  {
    // Update the stored error only if the previous error was queried
    if (device)
    {
      ErrorState& curError = device->error.get();

      if (curError.code == Error::None)
      {
        curError.code = code;
        curError.message = message;
      }

      // Print the error message in verbose mode
      if (device->isVerbose())
        std::cerr << "Error: " << message << std::endl;

      // Call the error callback function
      ErrorFunction errorFunc;
      void* errorUserPtr;

      {
        std::lock_guard<std::mutex> lock(device->mutex);
        errorFunc = device->errorFunc;
        errorUserPtr = device->errorUserPtr;
      }

      if (errorFunc)
        errorFunc(errorUserPtr, code, (code == Error::None) ? nullptr : message.c_str());
    }
    else
    {
      if (globalError.code == Error::None)
      {
        globalError.code = code;
        globalError.message = message;
      }
    }
  }

  Error Device::getError(Device* device, const char** outMessage)
  {
    // Return and clear the stored error code, but keep the error message so pointers to it will
    // remain valid until the next getError call
    if (device)
    {
      ErrorState& curError = device->error.get();
      const Error code = curError.code;
      if (outMessage)
        *outMessage = (code == Error::None) ? nullptr : curError.message.c_str();
      curError.code = Error::None;
      return code;
    }
    else
    {
      const Error code = globalError.code;
      if (outMessage)
        *outMessage = (code == Error::None) ? nullptr : globalError.message.c_str();
      globalError.code = Error::None;
      return code;
    }
  }

  void Device::setErrorFunction(ErrorFunction func, void* userPtr)
  {
    errorFunc = func;
    errorUserPtr = userPtr;
  }

  void Device::warning(const std::string& message)
  {
    OIDN_WARNING(message);
  }

  int Device::get1i(const std::string& name)
  {
    if (name == "numThreads")
      return numThreads;
    else if (name == "setAffinity")
      return setAffinity;
    else if (name == "verbose")
      return verbose;
    else if (name == "version")
      return OIDN_VERSION;
    else if (name == "versionMajor")
      return OIDN_VERSION_MAJOR;
    else if (name == "versionMinor")
      return OIDN_VERSION_MINOR;
    else if (name == "versionPatch")
      return OIDN_VERSION_PATCH;
    else
      throw Exception(Error::InvalidArgument, "unknown device parameter");
  }

  void Device::set1i(const std::string& name, int value)
  {
    if (name == "numThreads" && !isEnvVar("OIDN_NUM_THREADS"))
      numThreads = value;
    else if (name == "setAffinity" && !isEnvVar("OIDN_SET_AFFINITY"))
      setAffinity = value;
    else if (name == "verbose" && !isEnvVar("OIDN_VERBOSE"))
    {
      verbose = value;
      error.verbose = value;
    }
    else
      warning("unknown device parameter");

    dirty = true;
  }

  void Device::commit()
  {
    if (isCommitted())
      throw Exception(Error::InvalidOperation, "device can be committed only once");

    // Get the optimal thread affinities (except on Apple Silicon)
  #if !(defined(__APPLE__) && defined(OIDN_ARM64))
    if (setAffinity)
    {
      affinity = std::make_shared<ThreadAffinity>(1, verbose); // one thread per core
      if (affinity->getNumThreads() == 0)
        affinity.reset();
    }
  #endif

    // Create the task arena
    const int maxNumThreads = affinity ? affinity->getNumThreads() : tbb::this_task_arena::max_concurrency();
    numThreads = (numThreads > 0) ? min(numThreads, maxNumThreads) : maxNumThreads;
    arena = std::make_shared<tbb::task_arena>(numThreads);

    // Automatically set the thread affinities
    if (affinity)
      observer = std::make_shared<PinningObserver>(affinity, *arena);

    // Initialize the neural network runtime
  #if defined(OIDN_DNNL)
    dnnl_set_verbose(clamp(verbose - 2, 0, 2)); // unfortunately this is not per-device but global
    dnnlEngine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnlStream = dnnl::stream(dnnlEngine);
    tensorBlockSize = x64::mayiuse(x64::avx512_core) ? 16 : 8;
  #else
    tensorBlockSize = 1;
  #endif

    dirty = false;

    if (isVerbose())
      print();
  }

  void Device::checkCommitted()
  {
    if (dirty)
      throw Exception(Error::InvalidOperation, "changes to the device are not committed");
  }

  Ref<Buffer> Device::newBuffer(size_t byteSize)
  {
    return makeRef<CPUBuffer>(Ref<Device>(this), byteSize);
  }

  Ref<Buffer> Device::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<CPUBuffer>(Ref<Device>(this), ptr, byteSize);
  }

  Ref<Filter> Device::newFilter(const std::string& type)
  {
    if (isVerbose())
      std::cout << "Filter: " << type << std::endl;

    Ref<Filter> filter;

    if (type == "RT")
      filter = makeRef<RTFilter>(Ref<Device>(this));
    else if (type == "RTLightmap")
      filter = makeRef<RTLightmapFilter>(Ref<Device>(this));
    else
      throw Exception(Error::InvalidArgument, "unknown filter type");

    return filter;
  }

  Ref<ScratchBuffer> Device::newScratchBuffer(size_t byteSize)
  {
    auto scratchManager = scratchManagerWp.lock();
    if (!scratchManager)
      scratchManagerWp = scratchManager = std::make_shared<ScratchBufferManager>(this);
    return makeRef<ScratchBuffer>(scratchManager, byteSize);
  }

  void Device::print()
  {
    std::cout << std::endl;

    std::cout << "Intel(R) Open Image Denoise " << OIDN_VERSION_STRING << std::endl;
    std::cout << "  Compiler: " << getCompilerName() << std::endl;
    std::cout << "  Build   : " << getBuildName() << std::endl;
    std::cout << "  Platform: " << getPlatformName() << std::endl;

    std::cout << "  Targets :";
  #if defined(OIDN_X64)
    if (x64::mayiuse(x64::sse41))       std::cout << " SSE4.1";
    if (x64::mayiuse(x64::avx2))        std::cout << " AVX2";
    if (x64::mayiuse(x64::avx512_core)) std::cout << " AVX512";
  #elif defined(OIDN_ARM64)
    std::cout << " NEON";
  #endif
    std::cout << " (supported)" << std::endl;
    std::cout << "            ";
  #if defined(OIDN_X64)
    std::cout << "SSE4.1 AVX2 AVX512";
  #elif defined(OIDN_ARM64)
    std::cout << "NEON";
  #endif
    std::cout << " (compile time enabled)" << std::endl;
    
    std::cout << "  Neural  : ";
  #if defined(OIDN_DNNL)
    std::cout << "DNNL (oneDNN) " << DNNL_VERSION_MAJOR << "." <<
                                     DNNL_VERSION_MINOR << "." <<
                                     DNNL_VERSION_PATCH;
  #elif defined(OIDN_BNNS)
    std::cout << "BNNS";
  #endif
    std::cout << std::endl;

    std::cout << "  Tasking :";
    std::cout << " TBB" << TBB_VERSION_MAJOR << "." << TBB_VERSION_MINOR;
  #if TBB_INTERFACE_VERSION >= 12002
    std::cout << " TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << TBB_runtime_interface_version();
  #else
    std::cout << " TBB_header_interface_" << TBB_INTERFACE_VERSION << " TBB_lib_interface_" << tbb::TBB_runtime_interface_version();
  #endif

    std::cout << std::endl;

    std::cout << std::endl;
  }

} // namespace oidn
