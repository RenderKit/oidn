// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_external_buffer.h"

OIDN_NAMESPACE_BEGIN

  HIPExternalBuffer::HIPExternalBuffer(Engine* engine,
                                       ExternalMemoryTypeFlag fdType,
                                       int fd, size_t byteSize)
    : USMBuffer(engine)
  {
    if (fdType != ExternalMemoryTypeFlag::OpaqueFD)
      throw Exception(Error::InvalidArgument, "external memory type not supported by the device");

    hipExternalMemoryHandleDesc handleDesc{};
    handleDesc.type = hipExternalMemoryHandleTypeOpaqueFd;
    handleDesc.handle.fd = fd;
    handleDesc.size = byteSize;

    init(handleDesc);
  }

  HIPExternalBuffer::HIPExternalBuffer(Engine* engine,
                                       ExternalMemoryTypeFlag handleType,
                                       void* handle, const void* name, size_t byteSize)
    : USMBuffer(engine)
  {
    hipExternalMemoryHandleDesc handleDesc{};

    switch (handleType)
    {
    case ExternalMemoryTypeFlag::OpaqueWin32:
      handleDesc.type = hipExternalMemoryHandleTypeOpaqueWin32;
      break;
    case ExternalMemoryTypeFlag::OpaqueWin32KMT:
      handleDesc.type = hipExternalMemoryHandleTypeOpaqueWin32Kmt;
      break;
    case ExternalMemoryTypeFlag::D3D11Texture:
    case ExternalMemoryTypeFlag::D3D11Resource:
      handleDesc.type  = hipExternalMemoryHandleTypeD3D11Resource;
      handleDesc.flags = hipExternalMemoryDedicated;
      break;
    case ExternalMemoryTypeFlag::D3D11TextureKMT:
    case ExternalMemoryTypeFlag::D3D11ResourceKMT:
      handleDesc.type  = hipExternalMemoryHandleTypeD3D11ResourceKmt;
      handleDesc.flags = hipExternalMemoryDedicated;
      break;
    case ExternalMemoryTypeFlag::D3D12Heap:
      handleDesc.type = hipExternalMemoryHandleTypeD3D12Heap;
      break;
    case ExternalMemoryTypeFlag::D3D12Resource:
      handleDesc.type  = hipExternalMemoryHandleTypeD3D12Resource;
      handleDesc.flags = hipExternalMemoryDedicated;
      break;
    default:
      throw Exception(Error::InvalidArgument, "external memory type not supported by the device");
    }

    handleDesc.handle.win32.handle = handle;
    handleDesc.handle.win32.name = name;
    handleDesc.size = byteSize;

    init(handleDesc);
  }

  void HIPExternalBuffer::init(const hipExternalMemoryHandleDesc& handleDesc)
  {
    checkError(hipImportExternalMemory(&extMem, &handleDesc));

    void* devPtr = nullptr;
    hipExternalMemoryBufferDesc bufferDesc{};
    bufferDesc.offset = 0;
    bufferDesc.size   = handleDesc.size;
    bufferDesc.flags  = 0;
    checkError(hipExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufferDesc));

    ptr      = static_cast<char*>(devPtr);
    byteSize = handleDesc.size;
    shared   = true;
    storage  = Storage::Device;
  }

  HIPExternalBuffer::~HIPExternalBuffer()
  {
  #if HIP_VERSION_MAJOR >= 6
    hipFree(ptr);
  #endif
    hipDestroyExternalMemory(extMem);
  }

OIDN_NAMESPACE_END