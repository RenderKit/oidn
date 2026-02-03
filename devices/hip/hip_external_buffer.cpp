// Copyright 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_external_buffer.h"

OIDN_NAMESPACE_BEGIN

  HIPExternalBuffer::HIPExternalBuffer(Engine* engine,
                                       ExternalMemoryTypeFlags fdType,
                                       int fd, size_t byteSize)
    : USMBuffer(engine)
  {
    hipExternalMemoryHandleDesc handleDesc{};
    handleDesc.type = hipExternalMemoryHandleTypeOpaqueFd;
    handleDesc.handle.fd = fd;
    handleDesc.size = byteSize;

    if (fdType & ExternalMemoryTypeFlag::Dedicated)
    {
      handleDesc.flags = hipExternalMemoryDedicated;
      fdType ^= ExternalMemoryTypeFlag::Dedicated;
    }

    if (fdType != ExternalMemoryTypeFlag::OpaqueFD)
      throw Exception(Error::InvalidArgument, "external memory type not supported by the device");

    init(handleDesc);
  }

  HIPExternalBuffer::HIPExternalBuffer(Engine* engine,
                                       ExternalMemoryTypeFlags handleType,
                                       void* handle, const void* name, size_t byteSize)
    : USMBuffer(engine)
  {
    hipExternalMemoryHandleDesc handleDesc{};
    handleDesc.handle.win32.handle = handle;
    handleDesc.handle.win32.name = name;
    handleDesc.size = byteSize;

    if (handleType & ExternalMemoryTypeFlag::Dedicated)
    {
      handleDesc.flags = hipExternalMemoryDedicated;
      handleType ^= ExternalMemoryTypeFlag::Dedicated;
    }

    if (handleType == ExternalMemoryTypeFlag::OpaqueWin32)
      handleDesc.type = hipExternalMemoryHandleTypeOpaqueWin32;
    else if (handleType == ExternalMemoryTypeFlag::OpaqueWin32KMT)
      handleDesc.type = hipExternalMemoryHandleTypeOpaqueWin32Kmt;
    else if (handleType == ExternalMemoryTypeFlag::D3D11Texture ||
             handleType == ExternalMemoryTypeFlag::D3D11Resource)
    {
      handleDesc.type  = hipExternalMemoryHandleTypeD3D11Resource;
      handleDesc.flags = hipExternalMemoryDedicated; // mandatory
    }
    else if (handleType == ExternalMemoryTypeFlag::D3D11TextureKMT ||
             handleType == ExternalMemoryTypeFlag::D3D11ResourceKMT)
    {
      handleDesc.type  = hipExternalMemoryHandleTypeD3D11ResourceKmt;
      handleDesc.flags = hipExternalMemoryDedicated; // mandatory
    }
    else if (handleType == ExternalMemoryTypeFlag::D3D12Heap)
      handleDesc.type = hipExternalMemoryHandleTypeD3D12Heap;
    else if (handleType == ExternalMemoryTypeFlag::D3D12Resource)
    {
      handleDesc.type  = hipExternalMemoryHandleTypeD3D12Resource;
      handleDesc.flags = hipExternalMemoryDedicated; // mandatory
    }
    else
      throw Exception(Error::InvalidArgument, "external memory type not supported by the device");

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