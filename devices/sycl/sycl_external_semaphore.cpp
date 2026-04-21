// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_external_semaphore.h"

OIDN_NAMESPACE_BEGIN

  namespace oneapix = sycl::ext::oneapi::experimental;

  SYCLExternalSemaphore::SYCLExternalSemaphore(SYCLEngine* engine,
                                               ExternalSemaphoreTypeFlags fdType,
                                               int fd)
    : Semaphore(engine->getDevice()),
      type(fdType),
      engine(engine)
  {
  #if !defined(_WIN32)
    oneapix::external_semaphore_handle_type syclHandleType;

    if (fdType == ExternalSemaphoreTypeFlag::OpaqueFD)
      syclHandleType = oneapix::external_semaphore_handle_type::opaque_fd;
    else if (fdType == ExternalSemaphoreTypeFlag::TimelineSemaphoreFD)
      syclHandleType = oneapix::external_semaphore_handle_type::timeline_fd;
    else
      throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");

    oneapix::external_semaphore_descriptor<oneapix::resource_fd> desc{
      {fd},
      syclHandleType
    };

    extSem = oneapix::import_external_semaphore(
      desc,
      engine->getSYCLQueue().get_device(),
      engine->getSYCLQueue().get_context());
  #else
    throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");
  #endif
  }

  SYCLExternalSemaphore::SYCLExternalSemaphore(SYCLEngine* engine,
                                               ExternalSemaphoreTypeFlags handleType,
                                               void* handle, const void* name)
    : Semaphore(engine->getDevice()),
      type(handleType),
      engine(engine)
  {
  #if defined(_WIN32)
    oneapix::external_semaphore_handle_type syclHandleType;

    if (handleType == ExternalSemaphoreTypeFlag::OpaqueWin32)
      syclHandleType = oneapix::external_semaphore_handle_type::win32_nt_handle;
    else if (handleType == ExternalSemaphoreTypeFlag::D3D12Fence)
      syclHandleType = oneapix::external_semaphore_handle_type::win32_nt_dx12_fence;
    else if (handleType == ExternalSemaphoreTypeFlag::TimelineSemaphoreWin32)
      syclHandleType = oneapix::external_semaphore_handle_type::timeline_win32_nt_handle;
    else
      throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");

    if (handle != nullptr)
    {
      oneapix::external_semaphore_descriptor<oneapix::resource_win32_handle> desc{
        {handle},
        syclHandleType
      };
      extSem = oneapix::import_external_semaphore(
        desc,
        engine->getSYCLQueue().get_device(),
        engine->getSYCLQueue().get_context());
    }
    // FIXME: enable importing by name once it's supported by SYCL
    /*
    else if (name != nullptr)
    {
      oneapix::external_semaphore_descriptor<oneapix::resource_win32_name> desc{
        {name},
        syclHandleType
      };
      extSem = oneapix::import_external_semaphore(
        desc,
        engine->getSYCLQueue().get_device(),
        engine->getSYCLQueue().get_context());
    }
    */
    else
    {
      throw Exception(Error::InvalidArgument, "external semaphore handle and name are both null");
    }
  #else
    throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");
  #endif
  }

  SYCLExternalSemaphore::~SYCLExternalSemaphore()
  {
    oneapix::release_external_semaphore(
      extSem,
      engine->getSYCLQueue().get_device(),
      engine->getSYCLQueue().get_context());
  }

OIDN_NAMESPACE_END
