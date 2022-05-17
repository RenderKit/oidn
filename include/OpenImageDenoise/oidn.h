// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include "config.h"

OIDN_API_NAMESPACE_BEGIN

// -----------------------------------------------------------------------------
// Device
// -----------------------------------------------------------------------------

// Device types
typedef enum
{
  OIDN_DEVICE_TYPE_DEFAULT = 0, // select device automatically

  OIDN_DEVICE_TYPE_CPU  = 1, // CPU device
  OIDN_DEVICE_TYPE_SYCL = 2, // SYCL device
  OIDN_DEVICE_TYPE_CUDA = 3, // CUDA device
  OIDN_DEVICE_TYPE_HIP  = 4, // HIP device
} OIDNDeviceType;

// Error codes
typedef enum
{
  OIDN_ERROR_NONE                 = 0, // no error occurred
  OIDN_ERROR_UNKNOWN              = 1, // an unknown error occurred
  OIDN_ERROR_INVALID_ARGUMENT     = 2, // an invalid argument was specified
  OIDN_ERROR_INVALID_OPERATION    = 3, // the operation is not allowed
  OIDN_ERROR_OUT_OF_MEMORY        = 4, // not enough memory to execute the operation
  OIDN_ERROR_UNSUPPORTED_HARDWARE = 5, // the hardware (e.g. CPU) is not supported
  OIDN_ERROR_CANCELLED            = 6, // the operation was cancelled by the user
} OIDNError;

// Error callback function
typedef void (*OIDNErrorFunction)(void* userPtr, OIDNError code, const char* message);

// Device handle
typedef struct OIDNDeviceImpl* OIDNDevice;

// Creates a new device.
OIDN_API OIDNDevice oidnNewDevice(OIDNDeviceType type);

// Creates a new SYCL device using a specified in-order SYCL queue.
OIDN_API OIDNDevice oidnNewDeviceSYCL(void* syclQueue);

// Creates a new CUDA device using a specified CUDA stream.
OIDN_API OIDNDevice oidnNewDeviceCUDA(void* cudaStream);

// Creates a new HIP device using a specified HIP stream.
OIDN_API OIDNDevice oidnNewDeviceHIP(void* hipStream);

// Retains the device (increments the reference count).
OIDN_API void oidnRetainDevice(OIDNDevice device);

// Releases the device (decrements the reference count).
OIDN_API void oidnReleaseDevice(OIDNDevice device);

// Sets a boolean parameter of the device.
OIDN_API void oidnSetDevice1b(OIDNDevice device, const char* name, bool value);

// Sets an integer parameter of the device.
OIDN_API void oidnSetDevice1i(OIDNDevice device, const char* name, int value);

// Gets a boolean parameter of the device.
OIDN_API bool oidnGetDevice1b(OIDNDevice device, const char* name);

// Gets an integer parameter of the device (e.g. "version").
OIDN_API int oidnGetDevice1i(OIDNDevice device, const char* name);

// Sets the error callback function of the device.
OIDN_API void oidnSetDeviceErrorFunction(OIDNDevice device, OIDNErrorFunction func, void* userPtr);

// Returns the first unqueried error code stored in the device for the current
// thread, optionally also returning a string message (if not NULL), and clears
// the stored error. Can be called with a NULL device as well to check why a
// device creation failed.
OIDN_API OIDNError oidnGetDeviceError(OIDNDevice device, const char** outMessage);

// Commits all previous changes to the device.
// Must be called before first using the device (e.g. creating filters).
OIDN_API void oidnCommitDevice(OIDNDevice device);

// Waits for all asynchronous operations running on the device to complete.
OIDN_API void oidnSyncDevice(OIDNDevice device);

// -----------------------------------------------------------------------------
// Buffer
// -----------------------------------------------------------------------------

// Formats for images and other data stored in buffers
typedef enum
{
  OIDN_FORMAT_UNDEFINED = 0,

  // 32-bit single-precision floating-point scalar and vector formats
  OIDN_FORMAT_FLOAT  = 1,
  OIDN_FORMAT_FLOAT2,
  OIDN_FORMAT_FLOAT3,
  OIDN_FORMAT_FLOAT4,

  // 16-bit half-precision floating-point scalar and vector formats
  OIDN_FORMAT_HALF  = 257,
  OIDN_FORMAT_HALF2,
  OIDN_FORMAT_HALF3,
  OIDN_FORMAT_HALF4,
} OIDNFormat;

// Storage modes for buffers
typedef enum
{
  OIDN_STORAGE_UNDEFINED = 0,
  OIDN_STORAGE_HOST      = 1, // stored on the host, accessible to both the host and device
  OIDN_STORAGE_DEVICE    = 2, // stored on the device, *not* accessible to the host
  OIDN_STORAGE_MANAGED   = 3, // automatically migrated between the host and device, accessible to both
} OIDNStorage;

// Access modes for mapping buffers
typedef enum
{
  OIDN_ACCESS_READ          = 0, // read-only access
  OIDN_ACCESS_WRITE         = 1, // write-only access
  OIDN_ACCESS_READ_WRITE    = 2, // read and write access
  OIDN_ACCESS_WRITE_DISCARD = 3, // write-only access, previous contents discarded
} OIDNAccess;

// Buffer handle
typedef struct OIDNBufferImpl* OIDNBuffer;

// Creates a new buffer accessible to both the host and device.
OIDN_API OIDNBuffer oidnNewBuffer(OIDNDevice device, size_t byteSize);

// Creates a new buffer with the specified storage mode.
OIDN_API OIDNBuffer oidnNewBufferWithStorage(OIDNDevice device, size_t byteSize, OIDNStorage storage);

// Creates a new shared buffer allocated and owned by the user and accessible to the device.
OIDN_API OIDNBuffer oidnNewSharedBuffer(OIDNDevice device, void* devPtr, size_t byteSize);

// Maps a region of the buffer to the host memory.
// If byteSize is 0, the maximum available amount of memory will be mapped.
OIDN_API void* oidnMapBuffer(OIDNBuffer buffer, OIDNAccess access, size_t byteOffset, size_t byteSize);

// Unmaps a region of the buffer.
// mappedPtr must be a pointer returned by a previous call to oidnMapBuffer.
OIDN_API void oidnUnmapBuffer(OIDNBuffer buffer, void* mappedPtr);

// Reads data from a region of the buffer to host memory.
OIDN_API void oidnReadBuffer(OIDNBuffer buffer, size_t byteOffset, size_t byteSize, void* dstHostPtr);

// Writes data to a region of the buffer from host memory.
OIDN_API void oidnWriteBuffer(OIDNBuffer buffer, size_t byteOffset, size_t byteSize, const void* srcHostPtr);

// Gets a pointer to the buffer data, which is accessible to the device but not necessarily to the host as well.
OIDN_API void* oidnGetBufferData(OIDNBuffer buffer);

// Gets the size of the buffer in bytes.
OIDN_API size_t oidnGetBufferSize(OIDNBuffer buffer);

// Retains the buffer (increments the reference count).
OIDN_API void oidnRetainBuffer(OIDNBuffer buffer);

// Releases the buffer (decrements the reference count).
OIDN_API void oidnReleaseBuffer(OIDNBuffer buffer);

// -----------------------------------------------------------------------------
// Filter
// -----------------------------------------------------------------------------

// Progress monitor callback function
typedef bool (*OIDNProgressMonitorFunction)(void* userPtr, double n);

// Filter handle
typedef struct OIDNFilterImpl* OIDNFilter;

// Creates a new filter of the specified type (e.g. "RT").
OIDN_API OIDNFilter oidnNewFilter(OIDNDevice device, const char* type);

// Retains the filter (increments the reference count).
OIDN_API void oidnRetainFilter(OIDNFilter filter);

// Releases the filter (decrements the reference count).
OIDN_API void oidnReleaseFilter(OIDNFilter filter);

// Sets an image parameter of the filter with data stored in a buffer.
// If bytePixelStride and/or byteRowStride are zero, these will be computed automatically.
OIDN_API void oidnSetFilterImage(OIDNFilter filter, const char* name,
                                 OIDNBuffer buffer, OIDNFormat format,
                                 size_t width, size_t height,
                                 size_t byteOffset,
                                 size_t bytePixelStride, size_t byteRowStride);

// Sets an image parameter of the filter with data owned by the user and accessible to the device.
// If bytePixelStride and/or byteRowStride are zero, these will be computed automatically.
OIDN_API void oidnSetSharedFilterImage(OIDNFilter filter, const char* name,
                                       void* devPtr, OIDNFormat format,
                                       size_t width, size_t height,
                                       size_t byteOffset,
                                       size_t bytePixelStride, size_t byteRowStride);

// Removes an image parameter of the filter that was previously set.
OIDN_API void oidnRemoveFilterImage(OIDNFilter filter, const char* name);

// Sets an opaque data parameter of the filter owned by the user and accessible to the host.
OIDN_API void oidnSetSharedFilterData(OIDNFilter filter, const char* name,
                                      void* hostPtr, size_t byteSize);

// Notifies the filter that the contents of an opaque data parameter has been changed.
OIDN_API void oidnUpdateFilterData(OIDNFilter filter, const char* name);

// Removes an opaque data parameter of the filter that was previously set.
OIDN_API void oidnRemoveFilterData(OIDNFilter filter, const char* name);

// Sets a boolean parameter of the filter.
OIDN_API void oidnSetFilter1b(OIDNFilter filter, const char* name, bool value);

// Gets a boolean parameter of the filter.
OIDN_API bool oidnGetFilter1b(OIDNFilter filter, const char* name);

// Sets an integer parameter of the filter.
OIDN_API void oidnSetFilter1i(OIDNFilter filter, const char* name, int value);

// Gets an integer parameter of the filter.
OIDN_API int oidnGetFilter1i(OIDNFilter filter, const char* name);

// Sets a float parameter of the filter.
OIDN_API void oidnSetFilter1f(OIDNFilter filter, const char* name, float value);

// Gets a float parameter of the filter.
OIDN_API float oidnGetFilter1f(OIDNFilter filter, const char* name);

// Sets the progress monitor callback function of the filter.
OIDN_API void oidnSetFilterProgressMonitorFunction(OIDNFilter filter, OIDNProgressMonitorFunction func, void* userPtr);

// Commits all previous changes to the filter.
// Must be called before first executing the filter.
OIDN_API void oidnCommitFilter(OIDNFilter filter);

// Executes the filter.
OIDN_API void oidnExecuteFilter(OIDNFilter filter);

// Executes the filter asynchronously.
OIDN_API void oidnExecuteFilterAsync(OIDNFilter filter);

OIDN_API_NAMESPACE_END
