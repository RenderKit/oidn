// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "device.h"
#include "rt_filter.h"
#include "rtlightmap_filter.h"

namespace oidn {

  thread_local Device::ErrorState Device::globalError;

  Device::Device()
  {
  #if defined(OIDN_ARCH_X64)
    if (!isISASupported(ISA::SSE41))
      throw Exception(Error::UnsupportedHardware, "SSE4.1 support is required at minimum");
  #endif

    // Get default values from environment variables
    if (getEnvVar("OIDN_VERBOSE", verbose))
      error.verbose = verbose;
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
    if (name == "verbose")
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
    if (name == "verbose")
    {
      if (!isEnvVar("OIDN_VERBOSE"))
      {
        verbose = value;
        error.verbose = value;
      }
      else if (verbose != value || error.verbose != value)
        warning("OIDN_VERBOSE environment variable overrides device parameter");
    }
    else
      warning("unknown device parameter");

    dirty = true;
  }

  void Device::commit()
  {
    if (isCommitted())
      throw Exception(Error::InvalidOperation, "device can be committed only once");

    if (isVerbose())
    {
      std::cout << std::endl;
      std::cout << "Intel(R) Open Image Denoise " << OIDN_VERSION_STRING << std::endl;
      std::cout << "  Compiler  : " << getCompilerName() << std::endl;
      std::cout << "  Build     : " << getBuildName() << std::endl;
      std::cout << "  OS        : " << getOSName() << std::endl;
    }

    init();

    if (isVerbose())
      std::cout << std::endl;

    dirty = false;
    committed = true;
  }

  void Device::checkCommitted()
  {
    if (dirty)
      throw Exception(Error::InvalidOperation, "changes to the device are not committed");
  }

  Ref<Filter> Device::newFilter(const std::string& type)
  {
    if (isVerbose())
      std::cout << "Filter: " << type << std::endl;

    Ref<Filter> filter;

    if (type == "RT")
      filter = makeRef<RTFilter>(this);
    else if (type == "RTLightmap")
      filter = makeRef<RTLightmapFilter>(this);
    else
      throw Exception(Error::InvalidArgument, "unknown filter type");

    return filter;
  }

  Ref<Buffer> Device::newBuffer(size_t byteSize, Storage storage)
  {
    return getEngine()->newBuffer(byteSize, storage);
  }

  Ref<Buffer> Device::newBuffer(void* ptr, size_t byteSize)
  {
    return getEngine()->newBuffer(ptr, byteSize);
  }

  Storage Device::getPointerStorage(const void* ptr)
  {
    return getEngine()->getPointerStorage(ptr);
  }

} // namespace oidn
