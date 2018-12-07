// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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
  thread_local const char* Device::threadErrorMessage = nullptr;

  Device::Device()
    : error(Error::None),
      errorMessage(nullptr)
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

  void Device::setError(Device* device, Error error, const char* errorMessage)
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
        *errorMessage = device->errorMessage;
      device->error = Error::None;
      device->errorMessage = nullptr;
      return error;
    }
    else
    {
      const Error error = threadError;
      if (errorMessage)
        *errorMessage = threadErrorMessage;
      threadError = Error::None;
      threadErrorMessage = nullptr;
      return error;
    }
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

    if (type == "Autoencoder")
      filter = makeRef<Autoencoder>(Ref<Device>(this));
    else
      throw Exception(Error::InvalidArgument, "unknown filter type");

    return filter;
  }

} // namespace oidn
