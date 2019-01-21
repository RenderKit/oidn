// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "device.h"
#include "autoencoder.h"

namespace oidn {

  thread_local Error Device::threadError = Error::None;
  thread_local std::string Device::threadErrorMessage;

  Device::Device()
  {
    if (!mayiuse(sse42))
      throw Exception(Error::UnsupportedHardware, "SSE4.2 support is required at minimum");
  }

  Device::~Device()
  {
    observer.reset();
  }

  void Device::setError(Device* device, Error error, const std::string& errorMessage)
  {
    // Update the stored error only if the previous error was queried
    if (device)
    {
      if (device->error == Error::None)
      {
        device->error = error;
        device->errorMessage = errorMessage;
      }

      // Call the error callback function
      if (device->errorFunc)
      {
        device->errorFunc(device->errorUserPtr,
                          error, (error == Error::None) ? nullptr : device->errorMessage.c_str());
      }
    }
    else
    {
      if (threadError == Error::None)
      {
        threadError = error;
        threadErrorMessage = errorMessage;
      }
    }
  }

  Error Device::getError(Device* device, const char** errorMessage)
  {
    // Return and clear the stored error code, but keep the error message so pointers to it will
    // remain valid until the next getError call
    if (device)
    {
      const Error error = device->error;
      if (errorMessage)
        *errorMessage = (error == Error::None) ? nullptr : device->errorMessage.c_str();
      device->error = Error::None;
      return error;
    }
    else
    {
      const Error error = threadError;
      if (errorMessage)
        *errorMessage = (error == Error::None) ? nullptr : threadErrorMessage.c_str();
      threadError = Error::None;
      return error;
    }
  }

  void Device::setErrorFunction(ErrorFunction func, void* userPtr)
  {
    errorFunc = func;
    errorUserPtr = userPtr;
  }

  int Device::get1i(const std::string& name)
  {
    if (name == "numThreads")
      return numThreads;
    else if (name == "setAffinity")
      return setAffinity;
    else if (name == "version")
      return OIDN_VERSION;
    else if (name == "versionMajor")
      return OIDN_VERSION_MAJOR;
    else if (name == "versionMinor")
      return OIDN_VERSION_MINOR;
    else if (name == "versionPatch")
      return OIDN_VERSION_PATCH;
    else
      throw Exception(Error::InvalidArgument, "invalid parameter");
  }

  void Device::set1i(const std::string& name, int value)
  {
    if (name == "numThreads")
      numThreads = value;
    else if (name == "setAffinity")
      setAffinity = value;
  }

  void Device::commit()
  {
    if (isCommitted())
      throw Exception(Error::InvalidOperation, "a device can be committed only once");

    // Get the optimal thread affinities
    if (setAffinity)
      affinity = std::make_shared<ThreadAffinity>(1); // one thread per core

    // Create the task arena
    const int maxNumThreads = setAffinity ? affinity->getNumThreads() : tbb::this_task_arena::max_concurrency();
    numThreads = (numThreads > 0) ? min(numThreads, maxNumThreads) : maxNumThreads;
    arena = std::make_shared<tbb::task_arena>(numThreads);

    // Automatically set the thread affinities
    if (setAffinity)
      observer = std::make_shared<PinningObserver>(affinity, *arena);
  }

  void Device::checkCommitted()
  {
    if (!isCommitted())
      throw Exception(Error::InvalidOperation, "the device is not committed");
  }

  Ref<Buffer> Device::newBuffer(size_t byteSize)
  {
    checkCommitted();
    return makeRef<Buffer>(Ref<Device>(this), byteSize);
  }

  Ref<Buffer> Device::newBuffer(void* ptr, size_t byteSize)
  {
    checkCommitted();
    return makeRef<Buffer>(Ref<Device>(this), ptr, byteSize);
  }

  Ref<Filter> Device::newFilter(const std::string& type)
  {
    checkCommitted();

    Ref<Filter> filter;

    if (type == "RT")
      filter = makeRef<RTFilter>(Ref<Device>(this));
    else
      throw Exception(Error::InvalidArgument, "unknown filter type");

    return filter;
  }

} // namespace oidn
