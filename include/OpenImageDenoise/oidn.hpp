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

#pragma once

#include <algorithm>
#include "oidn.h"

namespace oidn {

  // ---------------------------------------------------------------------------
  // Buffer
  // ---------------------------------------------------------------------------

  // Formats for images and other data stored in buffers
  enum class Format
  {
    Undefined = OIDN_FORMAT_UNDEFINED,

    Float  = OIDN_FORMAT_FLOAT,
    Float2 = OIDN_FORMAT_FLOAT2,
    Float3 = OIDN_FORMAT_FLOAT3,
    Float4 = OIDN_FORMAT_FLOAT4,
  };

  // Access modes for mapping buffers
  enum class Access
  {
    Read         = OIDN_ACCESS_READ,          // read-only access
    Write        = OIDN_ACCESS_WRITE,         // write-only access
    ReadWrite    = OIDN_ACCESS_READ_WRITE,    // read and write access
    WriteDiscard = OIDN_ACCESS_WRITE_DISCARD, // write-only access, previous contents discarded
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

    // Maps a region of the buffer to host memory.
    void map(Access access = Access::ReadWrite, size_t byteOffset = 0, size_t byteSize = 0)
    {
      oidnMapBuffer(handle, (OIDNAccess)access, byteOffset, byteSize);
    }

    // Unmaps a region of the buffer.
    void unmap(void* mappedPtr)
    {
      oidnUnmapBuffer(handle, mappedPtr);
    }
  };


  // ---------------------------------------------------------------------------
  // Filter
  // ---------------------------------------------------------------------------

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

    // Sets an image parameter of the filter (stored in a buffer).
    void setImage(const char* name,
                  const BufferRef& buffer, Format format,
                  size_t width, size_t height,
                  size_t byteOffset = 0, size_t byteItemStride = 0, size_t byteRowStride = 0)
    {
      oidnSetFilterImage(handle, name,
                         buffer.getHandle(), (OIDNFormat)format,
                         width, height,
                         byteOffset, byteItemStride, byteRowStride);
    }

    // Sets an image parameter of the filter (owned by the user).
    void setImage(const char* name,
                  void* ptr, Format format,
                  size_t width, size_t height,
                  size_t byteOffset = 0, size_t byteItemStride = 0, size_t byteRowStride = 0)
    {
      oidnSetSharedFilterImage(handle, name,
                               ptr, (OIDNFormat)format,
                               width, height,
                               byteOffset, byteItemStride, byteRowStride);
    }

    // Sets a boolean parameter of the filter.
    void set(const char* name, bool value)
    {
      oidnSetFilter1b(handle, name, value);
    }

    // Sets an integer parameter of the filter.
    void set(const char* name, int value)
    {
      oidnSetFilter1i(handle, name, value);
    }

    // Commits all previous changes to the filter.
    void commit()
    {
      oidnCommitFilter(handle);
    }

    // Executes the filter.
    void execute()
    {
      oidnExecuteFilter(handle);
    }
  };


  // ---------------------------------------------------------------------------
  // Device
  // ---------------------------------------------------------------------------

  // Open Image Denoise device types
  enum class DeviceType
  {
    Default = OIDN_DEVICE_TYPE_DEFAULT,

    CPU = OIDN_DEVICE_TYPE_CPU,
  };

  // Error codes
  enum class Error
  {
    None                = OIDN_ERROR_NONE,
    Unknown             = OIDN_ERROR_UNKNOWN,
    InvalidArgument     = OIDN_ERROR_INVALID_ARGUMENT,
    InvalidOperation    = OIDN_ERROR_INVALID_OPERATION,
    OutOfMemory         = OIDN_ERROR_OUT_OF_MEMORY,
    UnsupportedHardware = OIDN_ERROR_UNSUPPORTED_HARDWARE,
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

    // Sets a boolean parameter of the device.
    void set(const char* name, bool value)
    {
      oidnSetDevice1b(handle, name, value);
    }

    // Sets an integer parameter of the device.
    void set(const char* name, int value)
    {
      oidnSetDevice1i(handle, name, value);
    }

    // Gets a parameter of the device.
    template<typename T>
    T get(const char* name);

    // Returns the first unqueried error code stored for the device, optionally
    // also returning a string message (if not null), and clears the stored error.
    Error getError(const char** message = nullptr)
    {
      return (Error)oidnGetDeviceError(handle, message);
    }

    // Commits all previous changes to the device.
    void commit()
    {
      oidnCommitDevice(handle);
    }

    // Creates a new buffer (data allocated and owned by the device).
    BufferRef newBuffer(size_t byteSize)
    {
      return oidnNewBuffer(handle, byteSize);
    }

    // Creates a new shared buffer (data allocated and owned by the user).
    BufferRef newBuffer(void* ptr, size_t byteSize)
    {
      return oidnNewSharedBuffer(handle, ptr, byteSize);
    }

    // Creates a new filter of the specified type.
    FilterRef newFilter(const char* type)
    {
      return oidnNewFilter(handle, type);
    }
  };

  // Gets a boolean parameter of the device.
  template<>
  inline bool DeviceRef::get(const char* name)
  {
    return oidnGetDevice1b(handle, name);
  }

  // Gets an integer parameter of the device (e.g., "version").
  template<>
  inline int DeviceRef::get(const char* name)
  {
    return oidnGetDevice1i(handle, name);
  }

  // Creates a new Open Image Denoise device.
  inline DeviceRef newDevice(DeviceType type = DeviceType::Default)
  {
    return DeviceRef(oidnNewDevice((OIDNDeviceType)type));
  }

} // namespace oidn
