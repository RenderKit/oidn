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

#ifdef _WIN32
#  define OIDN_API extern "C" __declspec(dllexport)
#else
#  define OIDN_API extern "C" __attribute__ ((visibility ("default")))
#endif

#define OIDN_TRY \
  std::lock_guard<std::mutex> apiLock(apiMutex); \
  try {

#define OIDN_CATCH(obj) \
  } catch (Exception& e) {                                                                          \
    Device::setError(obj ? obj->getDevice() : nullptr, e.code(), e.what());                         \
  } catch (std::bad_alloc&) {                                                                       \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::OutOfMemory, "out of memory");        \
  } catch (mkldnn::error& e) {                                                                      \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::Unknown, e.message);                  \
  } catch (std::exception& e) {                                                                     \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::Unknown, e.what());                   \
  } catch (...) {                                                                                   \
    Device::setError(obj ? obj->getDevice() : nullptr, Error::Unknown, "unknown exception caught"); \
  }

#include "device.h"
#include "filter.h"
#include <mutex>

namespace oidn {

  static std::mutex apiMutex;

  namespace
  {
    __forceinline void verifyHandle(void* handle)
    {
      if (handle == nullptr)
        throw Exception(Error::InvalidArgument, "invalid handle");
    }

    template<typename T>
    __forceinline void retainObject(T* obj)
    {
      if (obj)
      {
        obj->incRef();
      }
      else
      {
        OIDN_TRY
          verifyHandle(obj);
        OIDN_CATCH(obj)
      }
    }

    template<typename T>
    __forceinline void releaseObject(T* obj)
    {
      if (obj == nullptr || obj->decRefKeep() == 0)
      {
        OIDN_TRY
          verifyHandle(obj);
          obj->destroy();
        OIDN_CATCH(obj)
      }
    }
  }

  OIDN_API OIDNDevice oidnNewDevice(OIDNDeviceType type)
  {
    Ref<Device> device = nullptr;
    OIDN_TRY
      if (type == OIDN_DEVICE_TYPE_CPU || type == OIDN_DEVICE_TYPE_DEFAULT)
        device = makeRef<Device>();
      else
        throw Exception(Error::InvalidArgument, "invalid device type");
    OIDN_CATCH(device)
    return (OIDNDevice)device.detach();
  }

  OIDN_API void oidnRetainDevice(OIDNDevice hDevice)
  {
    Device* device = (Device*)hDevice;
    retainObject(device);
  }

  OIDN_API void oidnReleaseDevice(OIDNDevice hDevice)
  {
    Device* device = (Device*)hDevice;
    releaseObject(device);
  }

  OIDN_API bool oidnGetDevice1b(OIDNDevice hDevice, const char* name)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      verifyHandle(hDevice);
      return device->get1i(name);
    OIDN_CATCH(device)
    return false;
  }

  OIDN_API int oidnGetDevice1i(OIDNDevice hDevice, const char* name)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      verifyHandle(hDevice);
      return device->get1i(name);
    OIDN_CATCH(device)
    return 0;
  }

  OIDN_API void oidnSetDevice1b(OIDNDevice hDevice, const char* name, bool value)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      verifyHandle(hDevice);
      device->set1i(name, value);
    OIDN_CATCH(device)
  }

  OIDN_API void oidnSetDevice1i(OIDNDevice hDevice, const char* name, int value)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      verifyHandle(hDevice);
      device->set1i(name, value);
    OIDN_CATCH(device)
  }

  OIDN_API OIDNError oidnGetDeviceError(OIDNDevice hDevice, const char** message)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      return (OIDNError)Device::getError(device, message);
    OIDN_CATCH(device)
    if (message) *message = "";
    return OIDN_ERROR_UNKNOWN;
  }

  OIDN_API void oidnCommitDevice(OIDNDevice hDevice)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      verifyHandle(hDevice);
      device->commit();
    OIDN_CATCH(device)
  }

  OIDN_API OIDNBuffer oidnNewBuffer(OIDNDevice hDevice, size_t byteSize)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      verifyHandle(hDevice);
      Ref<Buffer> buffer = device->newBuffer(byteSize);
      return (OIDNBuffer)buffer.detach();
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API OIDNBuffer oidnNewSharedBuffer(OIDNDevice hDevice, void* ptr, size_t byteSize)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      verifyHandle(hDevice);
      Ref<Buffer> buffer = device->newBuffer(ptr, byteSize);
      return (OIDNBuffer)buffer.detach();
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API void oidnRetainBuffer(OIDNBuffer hBuffer)
  {
    Buffer* buffer = (Buffer*)hBuffer;
    retainObject(buffer);
  }

  OIDN_API void oidnReleaseBuffer(OIDNBuffer hBuffer)
  {
    Buffer* buffer = (Buffer*)hBuffer;
    releaseObject(buffer);
  }

  OIDN_API void* oidnMapBuffer(OIDNBuffer hBuffer, OIDNAccess access, size_t byteOffset, size_t byteSize)
  {
    Buffer* buffer = (Buffer*)hBuffer;
    OIDN_TRY
      verifyHandle(hBuffer);
      return buffer->map(byteOffset, byteSize);
    OIDN_CATCH(buffer)
    return nullptr;
  }

  OIDN_API void oidnUnmapBuffer(OIDNBuffer hBuffer, void* mappedPtr)
  {
    Buffer* buffer = (Buffer*)hBuffer;
    OIDN_TRY
      verifyHandle(hBuffer);
      return buffer->unmap(mappedPtr);
    OIDN_CATCH(buffer)
  }

  OIDN_API OIDNFilter oidnNewFilter(OIDNDevice hDevice, const char* type)
  {
    Device* device = (Device*)hDevice;
    OIDN_TRY
      verifyHandle(hDevice);
      Ref<Filter> filter = device->newFilter(type);
      return (OIDNFilter)filter.detach();
    OIDN_CATCH(device)
    return nullptr;
  }

  OIDN_API void oidnRetainFilter(OIDNFilter hFilter)
  {
    Filter* filter = (Filter*)hFilter;
    retainObject(filter);
  }

  OIDN_API void oidnReleaseFilter(OIDNFilter hFilter)
  {
    Filter* filter = (Filter*)hFilter;
    releaseObject(filter);
  }

  OIDN_API void oidnSetFilterImage(OIDNFilter hFilter, const char* name,
                                   OIDNBuffer hBuffer, OIDNFormat format,
                                   size_t width, size_t height,
                                   size_t byteOffset, size_t byteItemStride, size_t byteRowStride)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      verifyHandle(hFilter);
      verifyHandle(hBuffer);
      Ref<Buffer> buffer = (Buffer*)hBuffer;
      Image data(buffer, (Format)format, (int)width, (int)height, byteOffset, byteItemStride, byteRowStride);
      filter->setImage(name, data);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnSetSharedFilterImage(OIDNFilter hFilter, const char* name,
                                         void* ptr, OIDNFormat format,
                                         size_t width, size_t height,
                                         size_t byteOffset, size_t byteItemStride, size_t byteRowStride)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      verifyHandle(hFilter);
      Image data(ptr, (Format)format, (int)width, (int)height, byteOffset, byteItemStride, byteRowStride);
      filter->setImage(name, data);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnSetFilter1b(OIDNFilter hFilter, const char* name, bool value)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      verifyHandle(hFilter);
      filter->set1i(name, int(value));
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnSetFilter1i(OIDNFilter hFilter, const char* name, int value)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      verifyHandle(hFilter);
      filter->set1i(name, value);
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnCommitFilter(OIDNFilter hFilter)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      verifyHandle(hFilter);
      filter->commit();
    OIDN_CATCH(filter)
  }

  OIDN_API void oidnExecuteFilter(OIDNFilter hFilter)
  {
    Filter* filter = (Filter*)hFilter;
    OIDN_TRY
      verifyHandle(hFilter);
      filter->execute();
    OIDN_CATCH(filter)
  }

} // namespace oidn
