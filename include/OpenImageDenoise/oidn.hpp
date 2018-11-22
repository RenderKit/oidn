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

#pragma once

#include <algorithm>
#include "oidn.h"

namespace oidn {

  enum class DeviceType
  {
    CPU = OIDN_DEVICE_TYPE_CPU,
  };

  enum class Error
  {
    None                = OIDN_ERROR_NONE,
    Unknown             = OIDN_ERROR_UNKNOWN,
    InvalidArgument     = OIDN_ERROR_INVALID_ARGUMENT,
    InvalidOperation    = OIDN_ERROR_INVALID_OPERATION,
    OutOfMemory         = OIDN_ERROR_OUT_OF_MEMORY,
    UnsupportedHardware = OIDN_ERROR_UNSUPPORTED_HARDWARE,
  };

  enum class Format
  {
    Undefined = OIDN_FORMAT_UNDEFINED,

    Float  = OIDN_FORMAT_FLOAT,
    Float2 = OIDN_FORMAT_FLOAT2,
    Float3 = OIDN_FORMAT_FLOAT3,
  };

  // Buffer object with automatic reference counting
  class BufferRef
  {
  private:
    OIDNBuffer handle;

  public:
    BufferRef() : handle(nullptr) {}
    BufferRef(OIDNBuffer handle) : handle(handle) {}

    BufferRef(const BufferRef& other) : handle(other.handle)
    {
      if (handle)
        oidnRetainBuffer(handle);
    }

    BufferRef(BufferRef&& other) : handle(other.handle)
    {
      other.handle = nullptr;
    }

    BufferRef& operator =(const BufferRef& other)
    {
      if (&other != this)
      {
        if (other.handle)
          oidnRetainBuffer(other.handle);
        if (handle)
          oidnReleaseBuffer(handle);
        handle = other.handle;
      }
      return *this;
    }

    BufferRef& operator =(BufferRef&& other)
    {
      std::swap(handle, other.handle);
      return *this;
    }

    BufferRef& operator =(OIDNBuffer other)
    {
      if (other)
        oidnRetainBuffer(other);
      if (handle)
        oidnReleaseBuffer(handle);
      handle = other;
      return *this;
    }

    ~BufferRef()
    {
      if (handle)
        oidnReleaseBuffer(handle);
    }

    OIDNBuffer getHandle() const
    {
      return handle;
    }
  };

  // Filter object with automatic reference counting
  class FilterRef
  {
  private:
    OIDNFilter handle;

  public:
    FilterRef() : handle(nullptr) {}
    FilterRef(OIDNFilter handle) : handle(handle) {}

    FilterRef(const FilterRef& other) : handle(other.handle)
    {
      if (handle)
        oidnRetainFilter(handle);
    }

    FilterRef(FilterRef&& other) : handle(other.handle)
    {
      other.handle = nullptr;
    }

    FilterRef& operator =(const FilterRef& other)
    {
      if (&other != this)
      {
        if (other.handle)
          oidnRetainFilter(other.handle);
        if (handle)
          oidnReleaseFilter(handle);
        handle = other.handle;
      }
      return *this;
    }

    FilterRef& operator =(FilterRef&& other)
    {
      std::swap(handle, other.handle);
      return *this;
    }

    FilterRef& operator =(OIDNFilter other)
    {
      if (other)
        oidnRetainFilter(other);
      if (handle)
        oidnReleaseFilter(handle);
      handle = other;
      return *this;
    }

    ~FilterRef()
    {
      if (handle)
        oidnReleaseFilter(handle);
    }

    OIDNFilter getHandle() const
    {
      return handle;
    }

    void setData2D(const char* name,
                   const BufferRef& buffer, Format format,
                   size_t width, size_t height,
                   size_t byteOffset = 0, size_t byteStride = 0, size_t byteRowStride = 0)
    {
      oidnSetFilterData2D(handle, name,
                          buffer.getHandle(), (OIDNFormat)format,
                          width, height,
                          byteOffset, byteStride, byteRowStride);
    }

    void setData2D(const char* name,
                   void* ptr, Format format,
                   size_t width, size_t height,
                   size_t byteOffset = 0, size_t byteStride = 0, size_t byteRowStride = 0)
    {
      oidnSetSharedFilterData2D(handle, name,
                                ptr, (OIDNFormat)format,
                                width, height,
                                byteOffset, byteStride, byteRowStride);
    }

    void set1i(const char* name, int value)
    {
      oidnSetFilter1i(handle, name, value);
    }

    void commit()
    {
      oidnCommitFilter(handle);
    }

    void execute()
    {
      oidnExecuteFilter(handle);
    }
  };

  // Device object with automatic reference counting
  class DeviceRef
  {
  private:
    OIDNDevice handle;

  public:
    DeviceRef() : handle(nullptr) {}
    DeviceRef(OIDNDevice handle) : handle(handle) {}

    DeviceRef(const DeviceRef& other) : handle(other.handle)
    {
      if (handle)
        oidnRetainDevice(handle);
    }

    DeviceRef(DeviceRef&& other) : handle(other.handle)
    {
      other.handle = nullptr;
    }

    DeviceRef& operator =(const DeviceRef& other)
    {
      if (&other != this)
      {
        if (other.handle)
          oidnRetainDevice(other.handle);
        if (handle)
          oidnReleaseDevice(handle);
        handle = other.handle;
      }
      return *this;
    }

    DeviceRef& operator =(DeviceRef&& other)
    {
      std::swap(handle, other.handle);
      return *this;
    }

    DeviceRef& operator =(OIDNDevice other)
    {
      if (other)
        oidnRetainDevice(other);
      if (handle)
        oidnReleaseDevice(handle);
      handle = other;
      return *this;
    }

    ~DeviceRef()
    {
      if (handle)
        oidnReleaseDevice(handle);
    }

    OIDNDevice getHandle() const
    {
      return handle;
    }

    Error getError(const char** message = nullptr)
    {
      return (Error)oidnGetDeviceError(handle, message);
    }

    BufferRef newBuffer(size_t byteSize)
    {
      return oidnNewBuffer(handle, byteSize);
    }

    BufferRef newBuffer(void* ptr, size_t byteSize)
    {
      return oidnNewSharedBuffer(handle, ptr, byteSize);
    }

    FilterRef newFilter(const char* type)
    {
      return oidnNewFilter(handle, type);
    }
  };

  inline DeviceRef newDevice(DeviceType type)
  {
    return DeviceRef(oidnNewDevice((OIDNDeviceType)type));
  }

} // ::oidn
