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

namespace oidn {

  OIDN_API OIDNDevice oidnNewDevice()
  {
    Ref<Device> device = make_ref<Device>();
    return (OIDNDevice)device.detach();
  }

  OIDN_API void oidnRetainDevice(OIDNDevice hdevice)
  {
    Device* device = (Device*)hdevice;
    device->inc_ref();
  }

  OIDN_API void oidnReleaseDevice(OIDNDevice hdevice)
  {
    Device* device = (Device*)hdevice;
    device->dec_ref();
  }

  OIDN_API OIDNBuffer oidnNewSharedBuffer(OIDNDevice hdevice, void* ptr, size_t byteSize)
  {
    Device* device = (Device*)hdevice;
    Ref<Buffer> buffer = device->new_buffer(ptr, byteSize);
    return (OIDNBuffer)buffer.detach();
  }

  OIDN_API void oidnRetainBuffer(OIDNBuffer hbuffer)
  {
    Buffer* buffer = (Buffer*)hbuffer;
    buffer->inc_ref();
  }

  OIDN_API void oidnReleaseBuffer(OIDNBuffer hbuffer)
  {
    Buffer* buffer = (Buffer*)hbuffer;
    buffer->dec_ref();
  }

  OIDN_API OIDNFilter oidnNewFilter(OIDNDevice hdevice, OIDNFilterType type)
  {
    Device* device = (Device*)hdevice;
    Ref<Filter> filter = device->new_filter((FilterType)type);
    return (OIDNFilter)filter.detach();
  }

  OIDN_API void oidnRetainFilter(OIDNFilter hfilter)
  {
    Filter* filter = (Filter*)hfilter;
    filter->inc_ref();
  }

  OIDN_API void oidnReleaseFilter(OIDNFilter hfilter)
  {
    Filter* filter = (Filter*)hfilter;
    filter->dec_ref();
  }

  OIDN_API void oidnSetFilterBuffer2D(OIDNFilter hfilter, OIDNBufferType type, unsigned int slot,
                                      OIDNFormat format,
                                      OIDNBuffer hbuffer, size_t byteOffset, size_t byteStride,
                                      size_t width, size_t height)
  {
    Filter* filter = (Filter*)hfilter;
    Ref<Buffer> buffer = (Buffer*)hbuffer;
    BufferView2D view(buffer, byteOffset, (int)byteStride, (int)width, (int)height, (Format)format);
    filter->set_buffer((BufferType)type, slot, view);
  }

  OIDN_API void oidnSetSharedFilterBuffer2D(OIDNFilter hfilter, OIDNBufferType type, unsigned int slot,
                                            OIDNFormat format,
                                            const void* ptr, size_t byteOffset, size_t byteStride,
                                            size_t width, size_t height)
  {
    Filter* filter = (Filter*)hfilter;
    size_t byteSize = width * height * byteStride;
    Ref<Buffer> buffer = filter->get_device()->new_buffer((char*)ptr + byteOffset, byteSize);
    BufferView2D view(buffer, 0, (int)byteStride, (int)width, (int)height, (Format)format);
    filter->set_buffer((BufferType)type, slot, view);
  }

  OIDN_API void oidnCommitFilter(OIDNFilter hfilter)
  {
    Filter* filter = (Filter*)hfilter;
    filter->commit();
  }

  OIDN_API void oidnExecuteFilter(OIDNFilter hfilter)
  {
    Filter* filter = (Filter*)hfilter;
    filter->execute();
  }

} // ::oidn
