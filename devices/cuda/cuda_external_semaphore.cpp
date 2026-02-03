// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_external_semaphore.h"

OIDN_NAMESPACE_BEGIN

  CUDAExternalSemaphore::CUDAExternalSemaphore(Engine* engine,
                                               ExternalSemaphoreTypeFlags fdType,
                                               int fd)
    : Semaphore(engine->getDevice()),
      type(fdType)
  {
    cudaExternalSemaphoreHandleDesc handleDesc{};
    handleDesc.handle.fd = fd;

    if (fdType == ExternalSemaphoreTypeFlag::OpaqueFD)
      handleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
    else if (fdType == ExternalSemaphoreTypeFlag::TimelineSemaphoreFD)
      handleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    else
      throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");

    init(handleDesc);
  }

  CUDAExternalSemaphore::CUDAExternalSemaphore(Engine* engine,
                                               ExternalSemaphoreTypeFlags handleType,
                                               void* handle, const void* name)
    : Semaphore(engine->getDevice()),
      type(handleType)
  {
    cudaExternalSemaphoreHandleDesc handleDesc{};
    handleDesc.handle.win32.handle = handle;
    handleDesc.handle.win32.name = name;

    if (handleType == ExternalSemaphoreTypeFlag::OpaqueWin32)
      handleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
    else if (handleType == ExternalSemaphoreTypeFlag::OpaqueWin32KMT)
      handleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    else if (handleType == ExternalSemaphoreTypeFlag::D3D11Fence)
      handleDesc.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
    else if (handleType == ExternalSemaphoreTypeFlag::D3D12Fence)
      handleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
    else if (handleType == ExternalSemaphoreTypeFlag::KeyedMutex)
      handleDesc.type = cudaExternalSemaphoreHandleTypeKeyedMutex;
    else if (handleType == ExternalSemaphoreTypeFlag::KeyedMutexKMT)
      handleDesc.type = cudaExternalSemaphoreHandleTypeKeyedMutexKmt;
    else if (handleType == ExternalSemaphoreTypeFlag::TimelineSemaphoreWin32)
      handleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    else
      throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");

    init(handleDesc);
  }

  void CUDAExternalSemaphore::init(const cudaExternalSemaphoreHandleDesc& handleDesc)
  {
    checkError(cudaImportExternalSemaphore(&extSem, &handleDesc));
  }

  CUDAExternalSemaphore::~CUDAExternalSemaphore()
  {
    cudaDestroyExternalSemaphore(extSem);
  }

OIDN_NAMESPACE_END