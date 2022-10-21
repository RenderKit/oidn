// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include "oidn.h"

OIDN_NAMESPACE_BEGIN

  // ---------------------------------------------------------------------------
  // Buffer
  // ---------------------------------------------------------------------------

  // Formats for images and other data stored in buffers
  enum class Format
  {
    Undefined = OIDN_FORMAT_UNDEFINED,

    // 32-bit single-precision floating-point scalar and vector formats
    Float  = OIDN_FORMAT_FLOAT,
    Float2 = OIDN_FORMAT_FLOAT2,
    Float3 = OIDN_FORMAT_FLOAT3,
    Float4 = OIDN_FORMAT_FLOAT4,

    // 16-bit half-precision floating-point scalar and vector formats
    Half  = OIDN_FORMAT_HALF,
    Half2 = OIDN_FORMAT_HALF2,
    Half3 = OIDN_FORMAT_HALF3,
    Half4 = OIDN_FORMAT_HALF4,
  };

  // Storage modes for buffers
  enum class Storage
  {
    Undefined = OIDN_STORAGE_UNDEFINED,
    Host      = OIDN_STORAGE_HOST,      // stored on the host, accessible to both the host and device
    Device    = OIDN_STORAGE_DEVICE,    // stored on the device, *not* accessible to the host
    Managed   = OIDN_STORAGE_MANAGED,   // automatically migrated between the host and device, accessible to both
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

    BufferRef(BufferRef&& other) noexcept : handle(other.handle)
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

    BufferRef& operator =(BufferRef&& other) noexcept
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

    operator bool() const
    {
      return handle != nullptr;
    }

    // Gets the size of the buffer in bytes.
    size_t getSize() const
    {
      return oidnGetBufferSize(handle);
    }

    // Gets a pointer to the buffer data, which is accessible to the device but not necessarily to the host as well.
    void* getData() const
    {
      return oidnGetBufferData(handle);
    }

    // Maps a region of the buffer to the host memory.
    // If byteSize is 0, the maximum available amount of memory will be mapped.
    void* map(Access access = Access::ReadWrite, size_t byteOffset = 0, size_t byteSize = 0) const
    {
      return oidnMapBuffer(handle, (OIDNAccess)access, byteOffset, byteSize);
    }

    // Unmaps a region of the buffer.
    // mappedPtr must be a pointer returned by a previous call to map.
    void unmap(void* mappedPtr) const
    {
      oidnUnmapBuffer(handle, mappedPtr);
    }

    // Reads data from a region of the buffer to host memory.
    void read(size_t byteOffset, size_t byteSize, void* dstHostPtr)
    {
      oidnReadBuffer(handle, byteOffset, byteSize, dstHostPtr);
    }

    // Writes data to a region of the buffer from host memory.
    void write(size_t byteOffset, size_t byteSize, const void* srcHostPtr)
    {
      oidnWriteBuffer(handle, byteOffset, byteSize, srcHostPtr);
    }
  };

  // ---------------------------------------------------------------------------
  // Filter
  // ---------------------------------------------------------------------------

  // Progress monitor callback function
  typedef bool (*ProgressMonitorFunction)(void* userPtr, double n);

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

    FilterRef(FilterRef&& other) noexcept : handle(other.handle)
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

    FilterRef& operator =(FilterRef&& other) noexcept
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

    operator bool() const
    {
      return handle != nullptr;
    }

    // Sets an image parameter of the filter with data stored in a buffer.
    void setImage(const char* name,
                  const BufferRef& buffer, Format format,
                  size_t width, size_t height,
                  size_t byteOffset = 0,
                  size_t bytePixelStride = 0, size_t byteRowStride = 0)
    {
      oidnSetFilterImage(handle, name,
                         buffer.getHandle(), (OIDNFormat)format,
                         width, height,
                         byteOffset,
                         bytePixelStride, byteRowStride);
    }

    // Sets an image parameter of the filter with data owned by the user and accessible to the device.
    void setImage(const char* name,
                  void* devPtr, Format format,
                  size_t width, size_t height,
                  size_t byteOffset = 0,
                  size_t bytePixelStride = 0, size_t byteRowStride = 0)
    {
      oidnSetSharedFilterImage(handle, name,
                               devPtr, (OIDNFormat)format,
                               width, height,
                               byteOffset,
                               bytePixelStride, byteRowStride);
    }

    // Removes an image parameter of the filter that was previously set.
    void removeImage(const char* name)
    {
      oidnRemoveFilterImage(handle, name);
    }

    // Sets an opaque data parameter of the filter owned by the user and accessible to the host.
    void setData(const char* name,
                 void* hostPtr, size_t byteSize)
    {
      oidnSetSharedFilterData(handle, name,
                              hostPtr, byteSize);
    }

    // Notifies the filter that the contents of an opaque data parameter has been changed.
    void updateData(const char* name)
    {
      oidnUpdateFilterData(handle, name);
    }

    // Removes an opaque data parameter of the filter that was previously set.
    void removeData(const char* name)
    {
      oidnRemoveFilterData(handle, name);
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

    // Sets a float parameter of the filter.
    void set(const char* name, float value)
    {
      oidnSetFilter1f(handle, name, value);
    }

    // Gets a parameter of the filter.
    template<typename T>
    T get(const char* name) const;

    // Sets the progress monitor callback function of the filter.
    void setProgressMonitorFunction(ProgressMonitorFunction func, void* userPtr = nullptr)
    {
      oidnSetFilterProgressMonitorFunction(handle, (OIDNProgressMonitorFunction)func, userPtr);
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

    // Executes the filter asynchronously.
    void executeAsync()
    {
      oidnExecuteFilterAsync(handle);
    }

  #if defined(OIDN_DEVICE_SYCL) && defined(SYCL_LANGUAGE_VERSION)
    // Executes the SYCL filter asynchronously using the specified dependent events,
    // and returns an event for completion.
    sycl::event executeAsync(const std::vector<sycl::event>& depEvents)
    {
      sycl::event doneEvent;
      oidnExecuteSYCLFilterAsync(handle, depEvents.data(), int(depEvents.size()), &doneEvent);
      return doneEvent;
    }
  #endif
  };

  // Gets a boolean parameter of the filter.
  template<>
  inline bool FilterRef::get(const char* name) const
  {
    return oidnGetFilter1b(handle, name);
  }

  // Gets an integer parameter of the filter.
  template<>
  inline int FilterRef::get(const char* name) const
  {
    return oidnGetFilter1i(handle, name);
  }

  // Gets a float parameter of the filter.
  template<>
  inline float FilterRef::get(const char* name) const
  {
    return oidnGetFilter1f(handle, name);
  }

  // ---------------------------------------------------------------------------
  // Device
  // ---------------------------------------------------------------------------

  // Device types
  enum class DeviceType
  {
    Default = OIDN_DEVICE_TYPE_DEFAULT, // select device automatically

    CPU  = OIDN_DEVICE_TYPE_CPU,  // CPU device
    SYCL = OIDN_DEVICE_TYPE_SYCL, // SYCL device
    CUDA = OIDN_DEVICE_TYPE_CUDA, // CUDA device
    HIP  = OIDN_DEVICE_TYPE_HIP,  // HIP device
  };

  // Error codes
  enum class Error
  {
    None                = OIDN_ERROR_NONE,                 // no error occurred
    Unknown             = OIDN_ERROR_UNKNOWN,              // an unknown error occurred
    InvalidArgument     = OIDN_ERROR_INVALID_ARGUMENT,     // an invalid argument was specified
    InvalidOperation    = OIDN_ERROR_INVALID_OPERATION,    // the operation is not allowed
    OutOfMemory         = OIDN_ERROR_OUT_OF_MEMORY,        // not enough memory to execute the operation
    UnsupportedHardware = OIDN_ERROR_UNSUPPORTED_HARDWARE, // the hardware (e.g. CPU) is not supported
    Cancelled           = OIDN_ERROR_CANCELLED,            // the operation was cancelled by the user
  };

  // Error callback function
  typedef void (*ErrorFunction)(void* userPtr, Error code, const char* message);

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

    DeviceRef(DeviceRef&& other) noexcept : handle(other.handle)
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

    DeviceRef& operator =(DeviceRef&& other) noexcept
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

    operator bool() const
    {
      return handle != nullptr;
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
    T get(const char* name) const;

    // Sets the error callback function of the device.
    void setErrorFunction(ErrorFunction func, void* userPtr = nullptr)
    {
      oidnSetDeviceErrorFunction(handle, (OIDNErrorFunction)func, userPtr);
    }

    // Returns the first unqueried error code and clears the stored error.
    // Can be called for a null device as well to check why a device creation failed.
    Error getError()
    {
      return (Error)oidnGetDeviceError(handle, nullptr);
    }

    // Returns the first unqueried error code and string message, and clears the stored error.
    // Can be called for a null device as well to check why a device creation failed.
    Error getError(const char*& outMessage)
    {
      return (Error)oidnGetDeviceError(handle, &outMessage);
    }

    // Commits all previous changes to the device.
    // Must be called before first using the device (e.g. creating filters).
    void commit()
    {
      oidnCommitDevice(handle);
    }

    // Waits for all asynchronous operations running on the device to complete.
    void sync()
    {
      oidnSyncDevice(handle);
    }

    // Creates a new buffer accessible to both the host and device.
    BufferRef newBuffer(size_t byteSize) const
    {
      return oidnNewBuffer(handle, byteSize);
    }

    // Creates a new buffer with the specified storage mode.
    BufferRef newBuffer(size_t byteSize, Storage storage) const
    {
      return oidnNewBufferWithStorage(handle, byteSize, (OIDNStorage)storage);
    }

    // Creates a new shared buffer allocated and owned by the user.
    BufferRef newBuffer(void* ptr, size_t byteSize) const
    {
      return oidnNewSharedBuffer(handle, ptr, byteSize);
    }

    // Creates a new filter of the specified type (e.g. "RT").
    FilterRef newFilter(const char* type) const
    {
      return oidnNewFilter(handle, type);
    }
  };

  // Gets a boolean parameter of the device.
  template<>
  inline bool DeviceRef::get(const char* name) const
  {
    return oidnGetDevice1b(handle, name);
  }

  // Gets an integer parameter of the device (e.g. "version").
  template<>
  inline int DeviceRef::get(const char* name) const
  {
    return oidnGetDevice1i(handle, name);
  }

  // Creates a new Open Image Denoise device.
  inline DeviceRef newDevice(DeviceType type = DeviceType::Default)
  {
    return DeviceRef(oidnNewDevice((OIDNDeviceType)type));
  }

#if defined(OIDN_DEVICE_SYCL) && defined(SYCL_LANGUAGE_VERSION)
  // Creates a new Open Image Denoise device from the specified SYCL queue.
  inline DeviceRef newSYCLDevice(const sycl::queue& queue)
  {
    return DeviceRef(oidnNewSYCLDevice(&queue, 1));
  }

  // Creates a new Open Image Denoise device from the specified list of SYCL queues.
  // The queues should belong to different SYCL sub-devices (Xe-Stacks/Tiles) of the same SYCL root-device (GPU).
  inline DeviceRef newSYCLDevice(const std::vector<sycl::queue>& queues)
  {
    return DeviceRef(oidnNewSYCLDevice(queues.data(), int(queues.size())));
  }
#endif

#if defined(OIDN_DEVICE_CUDA)
  // Creates a new Open Image Denoise device from the specified CUDA stream.
  inline DeviceRef newCUDADevice(cudaStream_t stream)
  {
    return DeviceRef(oidnNewCUDADevice(stream));
  }
#endif

#if defined(OIDN_DEVICE_HIP)
  // Creates a new Open Image Denoise device from the specified HIP stream.
  inline DeviceRef newHIPDevice(hipStream_t stream)
  {
    return DeviceRef(oidnNewHIPDevice(stream));
  }
#endif

OIDN_NAMESPACE_END
