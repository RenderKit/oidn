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

namespace OIDN {

  enum class DeviceType
  {
    CPU = OIDN_DEVICE_TYPE_CPU,
  };

  enum class Format
  {
    UNDEFINED = OIDN_FORMAT_UNDEFINED,

    FLOAT  = OIDN_FORMAT_FLOAT,
    FLOAT2 = OIDN_FORMAT_FLOAT2,
    FLOAT3 = OIDN_FORMAT_FLOAT3,
  };

  // Buffer object with reference counting
  class Buffer
  {
  private:
    OIDNBuffer handle;

  public:
    Buffer() : handle(nullptr) {}
    Buffer(OIDNBuffer handle) : handle(handle) {}

    Buffer(const Buffer& other) : handle(other.handle)
    {
      if (handle)
        oidnRetainBuffer(handle);
    }

    Buffer(Buffer&& other) : handle(other.handle)
    {
      other.handle = nullptr;
    }

    Buffer& operator =(const Buffer& other)
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

    Buffer& operator =(Buffer&& other)
    {
      std::swap(handle, other.handle);
      return *this;
    }

    Buffer& operator =(OIDNBuffer other)
    {
      if (other)
        oidnRetainBuffer(other);
      if (handle)
        oidnReleaseBuffer(handle);
      handle = other;
      return *this;
    }

    ~Buffer()
    {
      if (handle)
        oidnReleaseBuffer(handle);
    }

    OIDNBuffer get() const
    {
      return handle;
    }
  };

  // Filter object with reference counting
  class Filter
  {
  private:
    OIDNFilter handle;

  public:
    Filter() : handle(nullptr) {}
    Filter(OIDNFilter handle) : handle(handle) {}

    Filter(const Filter& other) : handle(other.handle)
    {
      if (handle)
        oidnRetainFilter(handle);
    }

    Filter(Filter&& other) : handle(other.handle)
    {
      other.handle = nullptr;
    }

    Filter& operator =(const Filter& other)
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

    Filter& operator =(Filter&& other)
    {
      std::swap(handle, other.handle);
      return *this;
    }

    Filter& operator =(OIDNFilter other)
    {
      if (other)
        oidnRetainFilter(other);
      if (handle)
        oidnReleaseFilter(handle);
      handle = other;
      return *this;
    }

    ~Filter()
    {
      if (handle)
        oidnReleaseFilter(handle);
    }

    OIDNFilter get() const
    {
      return handle;
    }

    void setData2D(const char* name,
                   const Buffer& buffer, Format format,
                   size_t width, size_t height,
                   size_t byteOffset = 0, size_t byteStride = 0, size_t byteRowStride = 0)
    {
      oidnSetFilterData2D(handle, name,
                          buffer.get(), (OIDNFormat)format,
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

  // Device object with reference counting
  class Device
  {
  private:
    OIDNDevice handle;

  public:
    Device() : handle(nullptr) {}
    Device(OIDNDevice handle) : handle(handle) {}

    Device(const Device& other) : handle(other.handle)
    {
      if (handle)
        oidnRetainDevice(handle);
    }

    Device(Device&& other) : handle(other.handle)
    {
      other.handle = nullptr;
    }

    Device& operator =(const Device& other)
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

    Device& operator =(Device&& other)
    {
      std::swap(handle, other.handle);
      return *this;
    }

    Device& operator =(OIDNDevice other)
    {
      if (other)
        oidnRetainDevice(other);
      if (handle)
        oidnReleaseDevice(handle);
      handle = other;
      return *this;
    }

    ~Device()
    {
      if (handle)
        oidnReleaseDevice(handle);
    }

    OIDNDevice get() const
    {
      return handle;
    }

    Buffer newBuffer(void* ptr, size_t byteSize)
    {
      return oidnNewSharedBuffer(handle, ptr, byteSize);
    }

    Filter newFilter(const char* type)
    {
      return oidnNewFilter(handle, type);
    }
  };

  inline Device newDevice(DeviceType type)
  {
    return Device(oidnNewDevice((OIDNDeviceType)type));
  }

} // ::OIDN
