// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_external_semaphore.h"

OIDN_NAMESPACE_BEGIN

  HIPExternalSemaphore::HIPExternalSemaphore(Engine* engine,
                                             ExternalSemaphoreTypeFlags fdType,
                                             int fd)
    : Semaphore(engine->getDevice()),
      type(fdType)
  {
    // Not yet supported on Linux by HIP.
    throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");
  }

  HIPExternalSemaphore::HIPExternalSemaphore(Engine* engine,
                                             ExternalSemaphoreTypeFlags handleType,
                                             void* handle, const void* name)
    : Semaphore(engine->getDevice()),
      type(handleType)
  {
    hipExternalSemaphoreHandleDesc handleDesc{};
    handleDesc.handle.win32.handle = handle;
    handleDesc.handle.win32.name = name;

    if (handleType == ExternalSemaphoreTypeFlag::OpaqueWin32)
      handleDesc.type = hipExternalSemaphoreHandleTypeOpaqueWin32;
    else if (handleType == ExternalSemaphoreTypeFlag::OpaqueWin32KMT)
      handleDesc.type = hipExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    else if (handleType == ExternalSemaphoreTypeFlag::D3D11Fence)
      handleDesc.type = hipExternalSemaphoreHandleTypeD3D11Fence;
    else if (handleType == ExternalSemaphoreTypeFlag::D3D12Fence)
      handleDesc.type = hipExternalSemaphoreHandleTypeD3D12Fence;
    else if (handleType == ExternalSemaphoreTypeFlag::KeyedMutex)
      handleDesc.type = hipExternalSemaphoreHandleTypeKeyedMutex;
    else if (handleType == ExternalSemaphoreTypeFlag::KeyedMutexKMT)
      handleDesc.type = hipExternalSemaphoreHandleTypeKeyedMutexKmt;
    else if (handleType == ExternalSemaphoreTypeFlag::TimelineSemaphoreWin32)
      handleDesc.type = hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    else
      throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");

    init(handleDesc);
  }

  void HIPExternalSemaphore::init(const hipExternalSemaphoreHandleDesc& handleDesc)
  {
    checkError(hipImportExternalSemaphore(&extSem, &handleDesc));
  }

  HIPExternalSemaphore::~HIPExternalSemaphore()
  {
    hipDestroyExternalSemaphore(extSem);
  }

OIDN_NAMESPACE_END