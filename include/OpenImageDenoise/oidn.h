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

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#if defined(__cplusplus)
extern "C" {
#endif

#ifndef OIDN_API
#if defined(_WIN32) && !defined(OIDN_STATIC_LIB)
#  define OIDN_API __declspec(dllimport)
#else
#  define OIDN_API
#endif
#endif


// ---------------------------------------------------------------------------
// Device
// ---------------------------------------------------------------------------

// Open Image Denoise device types
typedef enum
# if __cplusplus >= 201103L
: uint32_t
#endif
{
  OIDN_DEVICE_TYPE_DEFAULT,

  OIDN_DEVICE_TYPE_CPU,
} OIDNDeviceType;

// Error codes
typedef enum
# if __cplusplus >= 201103L
: uint32_t
#endif
{
  OIDN_ERROR_NONE,
  OIDN_ERROR_UNKNOWN,
  OIDN_ERROR_INVALID_ARGUMENT,
  OIDN_ERROR_INVALID_OPERATION,
  OIDN_ERROR_OUT_OF_MEMORY,
  OIDN_ERROR_UNSUPPORTED_HARDWARE,
} OIDNError;

// Device handle
typedef struct OIDNDeviceImpl* OIDNDevice;

// Creates a new Open Image Denoise device.
OIDN_API OIDNDevice oidnNewDevice(OIDNDeviceType type);

// Retains the device (increments the reference count).
OIDN_API void oidnRetainDevice(OIDNDevice device);

// Releases the device (decrements the reference count).
OIDN_API void oidnReleaseDevice(OIDNDevice device);

// Returns the first unqueried error code stored for the device, optionally
// also returning a string message (if not null), and clears the stored error.
// If the device is null (e.g., the device creation failed), a thread-local
// error will be returned.
OIDN_API OIDNError oidnGetDeviceError(OIDNDevice device, const char** message);


// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

// Formats for images and other data stored in buffers
typedef enum
# if __cplusplus >= 201103L
: uint32_t
#endif
{
  OIDN_FORMAT_UNDEFINED,

  OIDN_FORMAT_FLOAT,
  OIDN_FORMAT_FLOAT2,
  OIDN_FORMAT_FLOAT3,
  OIDN_FORMAT_FLOAT4,
} OIDNFormat;

// Access modes for mapping buffers
typedef enum
# if __cplusplus >= 201103L
: uint32_t
#endif
{
  OIDN_ACCESS_READ,          // read-only access
  OIDN_ACCESS_WRITE,         // write-only access
  OIDN_ACCESS_READ_WRITE,    // read and write access
  OIDN_ACCESS_WRITE_DISCARD, // write-only access, previous contents discarded
} OIDNAccess;

// Buffer handle
typedef struct OIDNBufferImpl* OIDNBuffer;

// Creates a new buffer (data allocated and owned by the device).
OIDN_API OIDNBuffer oidnNewBuffer(OIDNDevice device, size_t byteSize);

// Creates a new shared buffer (data allocated and owned by the user).
OIDN_API OIDNBuffer oidnNewSharedBuffer(OIDNDevice device, void* ptr, size_t byteSize);

// Maps a region of the buffer to host memory.
// If byteSize is 0, the maximum available amount of memory will be mapped.
OIDN_API void* oidnMapBuffer(OIDNBuffer buffer, OIDNAccess access, size_t byteOffset, size_t byteSize);

// Unmaps a region of the buffer.
// mappedPtr must be a pointer returned by a previous call to oidnMapBuffer for the specified buffer.
OIDN_API void oidnUnmapBuffer(OIDNBuffer buffer, void* mappedPtr);

// Retains the buffer (increments the reference count).
OIDN_API void oidnRetainBuffer(OIDNBuffer buffer);

// Releases the buffer (decrements the reference count).
OIDN_API void oidnReleaseBuffer(OIDNBuffer buffer);


// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

// Filter handle
typedef struct OIDNFilterImpl* OIDNFilter;

// Creates a new filter of the specified type.
// Supported types:
//   RT - ray tracing denoising filter
OIDN_API OIDNFilter oidnNewFilter(OIDNDevice device, const char* type);

// Retains the filter (increments the reference count).
OIDN_API void oidnRetainFilter(OIDNFilter filter);

// Releases the filter (decrements the reference count).
OIDN_API void oidnReleaseFilter(OIDNFilter filter);

// Sets an image parameter of the filter (stored in a buffer).
// Supported parameters:
//   color  - input color to denoise
//   albedo - input albedo (optional)
//   normal - input normal (optional)
//   output - denoised output color
// All images must have FLOAT3 format and the same size.
// If byteItemStride and/or byteRowStride are zero, these will be computed
// automatically.
OIDN_API void oidnSetFilterImage(OIDNFilter filter, const char* name,
                                 OIDNBuffer buffer, OIDNFormat format,
                                 size_t width, size_t height,
                                 size_t byteOffset, size_t byteItemStride, size_t byteRowStride);

// Sets an image parameter of the filter (owned by the user).
OIDN_API void oidnSetSharedFilterImage(OIDNFilter filter, const char* name,
                                       void* ptr, OIDNFormat format,
                                       size_t width, size_t height,
                                       size_t byteOffset, size_t byteItemStride, size_t byteRowStride);

// Sets a boolean parameter of the filter.
// Supported parameters:
//   hdr  - the color image has high dynamic range (HDR), if non-zero (default is false)
//   srgb - the color image is encoded in sRGB or 2.2 gamma space, if non-zero (default is false)
OIDN_API void oidnSetFilter1b(OIDNFilter filter, const char* name, bool value);

// Sets an integer parameter of the filter.
OIDN_API void oidnSetFilter1i(OIDNFilter filter, const char* name, int value);

// Commits all previous changes to the filter.
// Must be called before execution.
OIDN_API void oidnCommitFilter(OIDNFilter filter);

// Executes the filter.
OIDN_API void oidnExecuteFilter(OIDNFilter filter);

#if defined(__cplusplus)
}
#endif
