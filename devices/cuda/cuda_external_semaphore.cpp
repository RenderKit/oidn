// Copyright 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_external_semaphore.h"

OIDN_NAMESPACE_BEGIN

  CUDAExternalSemaphore::CUDAExternalSemaphore(Engine* engine,
                                               ExternalSemaphoreTypeFlag fdType,
                                               int fd)
    : Semaphore(engine->getDevice()),
      type(fdType)
  {
    cudaExternalSemaphoreHandleDesc handleDesc{};

    switch (fdType)
    {
    case ExternalSemaphoreTypeFlag::OpaqueFD:
      handleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
      break;
    case ExternalSemaphoreTypeFlag::TimelineSemaphoreFD:
      handleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
      break;
    default:
      throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");
    }

    handleDesc.handle.fd = fd;

    init(handleDesc);
  }

  CUDAExternalSemaphore::CUDAExternalSemaphore(Engine* engine,
                                               ExternalSemaphoreTypeFlag handleType,
                                               void* handle, const void* name)
    : Semaphore(engine->getDevice()),
      type(handleType)
  {
    cudaExternalSemaphoreHandleDesc handleDesc{};

    switch (handleType)
    {
    case ExternalSemaphoreTypeFlag::OpaqueWin32:
      handleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
      break;
    case ExternalSemaphoreTypeFlag::OpaqueWin32KMT:
      handleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
      break;
    case ExternalSemaphoreTypeFlag::D3D11Fence:
      handleDesc.type = cudaExternalSemaphoreHandleTypeD3D11Fence;
      break;
    case ExternalSemaphoreTypeFlag::D3D12Fence:
      handleDesc.type = cudaExternalSemaphoreHandleTypeD3D12Fence;
      break;
    case ExternalSemaphoreTypeFlag::KeyedMutex:
      handleDesc.type = cudaExternalSemaphoreHandleTypeKeyedMutex;
      break;
    case ExternalSemaphoreTypeFlag::KeyedMutexKMT:
      handleDesc.type = cudaExternalSemaphoreHandleTypeKeyedMutexKmt;
      break;
    case ExternalSemaphoreTypeFlag::TimelineSemaphoreWin32:
      handleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
      break;
    default:
      throw Exception(Error::InvalidArgument, "external semaphore type not supported by the device");
    }

    handleDesc.handle.win32.handle = handle;
    handleDesc.handle.win32.name = name;

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