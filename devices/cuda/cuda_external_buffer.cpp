// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_external_buffer.h"

OIDN_NAMESPACE_BEGIN

  CUDAExternalBuffer::CUDAExternalBuffer(Engine* engine,
                                         ExternalMemoryTypeFlag fdType,
                                         int fd, size_t byteSize)
    : USMBuffer(engine)
  {
    if (fdType != ExternalMemoryTypeFlag::OpaqueFD)
      throw Exception(Error::InvalidArgument, "external memory type not supported by the device");

    cudaExternalMemoryHandleDesc handleDesc{};
    handleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    handleDesc.handle.fd = fd;
    handleDesc.size = byteSize;

    init(handleDesc);
  }

  CUDAExternalBuffer::CUDAExternalBuffer(Engine* engine,
                                         ExternalMemoryTypeFlag handleType,
                                         void* handle, const void* name, size_t byteSize)
    : USMBuffer(engine)
  {
    cudaExternalMemoryHandleDesc handleDesc{};

    switch (handleType)
    {
    case ExternalMemoryTypeFlag::OpaqueWin32:
      handleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
      break;
    case ExternalMemoryTypeFlag::OpaqueWin32KMT:
      handleDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
      break;
    case ExternalMemoryTypeFlag::D3D11Texture:
    case ExternalMemoryTypeFlag::D3D11Resource:
      handleDesc.type  = cudaExternalMemoryHandleTypeD3D11Resource;
      handleDesc.flags = cudaExternalMemoryDedicated;
      break;
    case ExternalMemoryTypeFlag::D3D11TextureKMT:
    case ExternalMemoryTypeFlag::D3D11ResourceKMT:
      handleDesc.type  = cudaExternalMemoryHandleTypeD3D11ResourceKmt;
      handleDesc.flags = cudaExternalMemoryDedicated;
      break;
    case ExternalMemoryTypeFlag::D3D12Heap:
      handleDesc.type = cudaExternalMemoryHandleTypeD3D12Heap;
      break;
    case ExternalMemoryTypeFlag::D3D12Resource:
      handleDesc.type  = cudaExternalMemoryHandleTypeD3D12Resource;
      handleDesc.flags = cudaExternalMemoryDedicated;
      break;
    default:
      throw Exception(Error::InvalidArgument, "external memory type not supported by the device");
    }

    handleDesc.handle.win32.handle = handle;
    handleDesc.handle.win32.name = name;
    handleDesc.size = byteSize;

    init(handleDesc);
  }

  void CUDAExternalBuffer::init(const cudaExternalMemoryHandleDesc& handleDesc)
  {
    checkError(cudaImportExternalMemory(&extMem, &handleDesc));

    void* devPtr = nullptr;
    cudaExternalMemoryBufferDesc bufferDesc{};
    bufferDesc.offset = 0;
    bufferDesc.size   = handleDesc.size;
    bufferDesc.flags  = 0;
    checkError(cudaExternalMemoryGetMappedBuffer(&devPtr, extMem, &bufferDesc));

    ptr      = static_cast<char*>(devPtr);
    byteSize = handleDesc.size;
    shared   = true;
    storage  = Storage::Device;
  }

  CUDAExternalBuffer::~CUDAExternalBuffer()
  {
    cudaFree(ptr);
    cudaDestroyExternalMemory(extMem);
  }

OIDN_NAMESPACE_END