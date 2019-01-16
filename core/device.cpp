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
    : error(Error::None)
  {
    if (!mayiuse(sse42))
      throw Exception(Error::UnsupportedHardware, "SSE4.2 support is required at minimum");

    affinity = std::make_shared<ThreadAffinity>(1); // one thread per core
    arena = std::make_shared<tbb::task_arena>(affinity->getNumThreads());
    observer = std::make_shared<PinningObserver>(affinity, *arena);
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
    // Return and clear the stored error
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

  int Device::get1i(const std::string& name)
  {
    if (name == "version")
      return OIDN_VERSION;
    else if (name == "versionMajor")
      return OIDN_VERSION_MAJOR;
    else if (name == "versionMinor")
      return OIDN_VERSION_MINOR;
    else if (name == "versionPatch")
      return OIDN_VERSION_PATCH;
    else
      throw Exception(Error::InvalidArgument, "unknown parameter");
  }

  Ref<Buffer> Device::newBuffer(size_t byteSize)
  {
    return makeRef<Buffer>(Ref<Device>(this), byteSize);
  }

  Ref<Buffer> Device::newBuffer(void* ptr, size_t byteSize)
  {
    return makeRef<Buffer>(Ref<Device>(this), ptr, byteSize);
  }

  Ref<Filter> Device::newFilter(const std::string& type)
  {
    Ref<Filter> filter;

    if (type == "RT")
      filter = makeRef<RTFilter>(Ref<Device>(this));
    else
      throw Exception(Error::InvalidArgument, "unknown filter type");

    return filter;
  }

} // namespace oidn
