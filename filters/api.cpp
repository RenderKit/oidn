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

#ifdef _WIN32
#  define OIDN_API extern "C" __declspec(dllexport)
#else
#  define OIDN_API extern "C" __attribute__ ((visibility ("default")))
#endif

#include "device.h"
#include "filter.h"
#include <mutex>

namespace oidn {

  static std::mutex api_mutex;

  OIDN_API OIDNDevice oidnNewDevice(OIDNDeviceType type)
  {
    std::lock_guard<std::mutex> lock(api_mutex);

    Ref<Device> device = nullptr;

    if (type == OIDN_DEVICE_TYPE_CPU)
      device = make_ref<Device>();

    return (OIDNDevice)device.detach();
  }

  OIDN_API void oidnRetainDevice(OIDNDevice hdevice)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Device* device = (Device*)hdevice;
    device->incRef();
  }

  OIDN_API void oidnReleaseDevice(OIDNDevice hdevice)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Device* device = (Device*)hdevice;
    device->decRef();
  }

  OIDN_API OIDNBuffer oidnNewSharedBuffer(OIDNDevice hdevice, void* ptr, size_t byteSize)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Device* device = (Device*)hdevice;
    Ref<Buffer> buffer = device->newBuffer(ptr, byteSize);
    return (OIDNBuffer)buffer.detach();
  }

  OIDN_API void oidnRetainBuffer(OIDNBuffer hbuffer)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Buffer* buffer = (Buffer*)hbuffer;
    buffer->incRef();
  }

  OIDN_API void oidnReleaseBuffer(OIDNBuffer hbuffer)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Buffer* buffer = (Buffer*)hbuffer;
    buffer->decRef();
  }

  OIDN_API OIDNFilter oidnNewFilter(OIDNDevice hdevice, const char* type)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Device* device = (Device*)hdevice;
    Ref<Filter> filter = device->newFilter(type);
    return (OIDNFilter)filter.detach();
  }

  OIDN_API void oidnRetainFilter(OIDNFilter hfilter)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Filter* filter = (Filter*)hfilter;
    filter->incRef();
  }

  OIDN_API void oidnReleaseFilter(OIDNFilter hfilter)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Filter* filter = (Filter*)hfilter;
    filter->decRef();
  }

  OIDN_API void oidnSetFilterData2D(OIDNFilter hfilter, const char* name,
                                    OIDNBuffer hbuffer, OIDNFormat format,
                                    size_t width, size_t height,
                                    size_t byteOffset, size_t byteStride, size_t byteRowStride)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Filter* filter = (Filter*)hfilter;
    Ref<Buffer> buffer = (Buffer*)hbuffer;
    Data2D data(buffer, (Format)format, (int)width, (int)height, byteOffset, byteStride, byteRowStride);
    filter->setData2D(name, data);
  }

  OIDN_API void oidnSetSharedFilterData2D(OIDNFilter hfilter, const char* name,
                                          void* ptr, OIDNFormat format,
                                          size_t width, size_t height,
                                          size_t byteOffset, size_t byteStride, size_t byteRowStride)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Filter* filter = (Filter*)hfilter;
    Data2D data(ptr, (Format)format, (int)width, (int)height, byteOffset, byteStride, byteRowStride);
    filter->setData2D(name, data);
  }

  OIDN_API void oidnSetFilter1i(OIDNFilter hfilter, const char* name, int value)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Filter* filter = (Filter*)hfilter;
    filter->set1i(name, value);
  }

  OIDN_API void oidnCommitFilter(OIDNFilter hfilter)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Filter* filter = (Filter*)hfilter;
    filter->commit();
  }

  OIDN_API void oidnExecuteFilter(OIDNFilter hfilter)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    Filter* filter = (Filter*)hfilter;
    filter->execute();
  }

} // ::oidn
