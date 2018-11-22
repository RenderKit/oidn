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

#define OIDN_TRY \
  std::lock_guard<std::mutex> apiLock(apiMutex); \
  try {

#define OIDN_CATCH(device) \
  } catch (Exception& e) {                                                \
    Device::setError(device, e.code(), e.what());                         \
  } catch (std::bad_alloc&) {                                             \
    Device::setError(device, Error::OutOfMemory, "out of memory");        \
  } catch (std::exception& e) {                                           \
    Device::setError(device, Error::Unknown, e.what());                   \
  } catch (...) {                                                         \
    Device::setError(device, Error::Unknown, "unknown exception caught"); \
  }

#include "device.h"
#include "filter.h"
#include <mutex>

namespace oidn {

  static std::mutex apiMutex;

  namespace
  {
    void verifyHandle(void* handle)
    {
      if (handle == nullptr)
        throw Exception(Error::InvalidArgument, "invalid handle");
    }
  }

  OIDN_API OIDNDevice oidnNewDevice(OIDNDeviceType type)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      if (type == OIDN_DEVICE_TYPE_CPU)
        device = makeRef<Device>();
      else
        throw Exception(Error::InvalidArgument, "invalid device type");
    OIDN_CATCH(device.get())
    return (OIDNDevice)device.detach();
  }

  OIDN_API void oidnRetainDevice(OIDNDevice hdevice)
  {
    Device* device = (Device*)hdevice;
    OIDN_TRY
      verifyHandle(hdevice);
      device->incRef();
    OIDN_CATCH(device)
  }

  OIDN_API void oidnReleaseDevice(OIDNDevice hdevice)
  {
    Device* device = (Device*)hdevice;
    OIDN_TRY
      verifyHandle(hdevice);
      device->decRef();
    OIDN_CATCH(device)
  }

  OIDN_API OIDNError oidnGetDeviceError(OIDNDevice hdevice, const char** message)
  {
    Device* device = (Device*)hdevice;
    OIDN_TRY
      return (OIDNError)Device::getError(device, message);
    OIDN_CATCH(device)
    if (message) *message = "";
    return OIDN_ERROR_UNKNOWN;
  }

  OIDN_API OIDNBuffer oidnNewBuffer(OIDNDevice hdevice, size_t byteSize)
  {
    Device* device = (Device*)hdevice;
    OIDN_TRY
      verifyHandle(hdevice);
      Ref<Buffer> buffer = device->newBuffer(byteSize);
      return (OIDNBuffer)buffer.detach();
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBuffer(OIDNDevice hdevice, void* ptr, size_t byteSize)
  {
    Device* device = (Device*)hdevice;
    OIDN_TRY
      verifyHandle(hdevice);
      Ref<Buffer> buffer = device->newBuffer(ptr, byteSize);
      return (OIDNBuffer)buffer.detach();
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API void oidnRetainBuffer(OIDNBuffer hbuffer)
  {
    Buffer* buffer = (Buffer*)hbuffer;
    OIDN_TRY
      verifyHandle(hbuffer);
      buffer->incRef();
    OIDN_CATCH(buffer->getDevice().get())
  }

  OIDN_API void oidnReleaseBuffer(OIDNBuffer hbuffer)
  {
    Buffer* buffer = (Buffer*)hbuffer;
    OIDN_TRY
      verifyHandle(hbuffer);
      buffer->decRef();
    OIDN_CATCH(buffer->getDevice().get())
  }

  OIDN_API OIDNFilter oidnNewFilter(OIDNDevice hdevice, const char* type)
  {
    Device* device = (Device*)hdevice;
    OIDN_TRY
      verifyHandle(hdevice);
      Ref<Filter> filter = device->newFilter(type);
      return (OIDNFilter)filter.detach();
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API void oidnRetainFilter(OIDNFilter hfilter)
  {
    Filter* filter = (Filter*)hfilter;
    OIDN_TRY
      verifyHandle(hfilter);
      filter->incRef();
    OIDN_CATCH(filter->getDevice().get())
  }

  OIDN_API void oidnReleaseFilter(OIDNFilter hfilter)
  {
    Filter* filter = (Filter*)hfilter;
    OIDN_TRY
      verifyHandle(hfilter);
      filter->decRef();
    OIDN_CATCH(filter->getDevice().get())
  }

  OIDN_API void oidnSetFilterImage(OIDNFilter hfilter, const char* name,
                                   OIDNBuffer hbuffer, OIDNFormat format,
                                   size_t width, size_t height,
                                   size_t byteOffset, size_t byteItemStride, size_t byteRowStride)
  {
    Filter* filter = (Filter*)hfilter;
    OIDN_TRY
      verifyHandle(hfilter);
      verifyHandle(hbuffer);
      Ref<Buffer> buffer = (Buffer*)hbuffer;
      Image data(buffer, (Format)format, (int)width, (int)height, byteOffset, byteItemStride, byteRowStride);
      filter->setImage(name, data);
    OIDN_CATCH(filter->getDevice().get())
  }

  OIDN_API void oidnSetSharedFilterImage(OIDNFilter hfilter, const char* name,
                                         void* ptr, OIDNFormat format,
                                         size_t width, size_t height,
                                         size_t byteOffset, size_t byteItemStride, size_t byteRowStride)
  {
    Filter* filter = (Filter*)hfilter;
    OIDN_TRY
      verifyHandle(hfilter);
      Image data(ptr, (Format)format, (int)width, (int)height, byteOffset, byteItemStride, byteRowStride);
      filter->setImage(name, data);
    OIDN_CATCH(filter->getDevice().get())
  }

  OIDN_API void oidnSetFilter1i(OIDNFilter hfilter, const char* name, int value)
  {
    Filter* filter = (Filter*)hfilter;
    OIDN_TRY
      verifyHandle(hfilter);
      filter->set1i(name, value);
    OIDN_CATCH(filter->getDevice().get())
  }

  OIDN_API void oidnCommitFilter(OIDNFilter hfilter)
  {
    Filter* filter = (Filter*)hfilter;
    OIDN_TRY
      verifyHandle(hfilter);
      filter->commit();
    OIDN_CATCH(filter->getDevice().get())
  }

  OIDN_API void oidnExecuteFilter(OIDNFilter hfilter)
  {
    Filter* filter = (Filter*)hfilter;
    OIDN_TRY
      verifyHandle(hfilter);
      filter->execute();
    OIDN_CATCH(filter->getDevice().get())
  }

} // ::oidn
