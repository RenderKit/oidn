// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_external_buffer.h"

OIDN_NAMESPACE_BEGIN

  SYCLExternalBuffer::SYCLExternalBuffer(SYCLEngine* engine,
                                         ExternalMemoryTypeFlag fdType,
                                         int fd, size_t byteSize)
    : USMBuffer(engine)
  {
    if (fdType != ExternalMemoryTypeFlag::DMABuf)
      throw Exception(Error::InvalidArgument, "external memory type not supported by the device");

    ze_external_memory_import_fd_t importDesc =
    {
      ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD,
      nullptr, // pNext
      ZE_EXTERNAL_MEMORY_TYPE_FLAG_DMA_BUF,
      fd
    };

    init(engine, &importDesc, byteSize);
  }

  SYCLExternalBuffer::SYCLExternalBuffer(SYCLEngine* engine,
                                         ExternalMemoryTypeFlag handleType,
                                         void* handle, const void* name, size_t byteSize)
    : USMBuffer(engine)
  {
    if (handleType != ExternalMemoryTypeFlag::OpaqueWin32)
      throw Exception(Error::InvalidArgument, "external memory type not supported by the device");

    ze_external_memory_import_win32_handle_t importDesc =
    {
      ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_WIN32,
      nullptr, // pNext
      ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_WIN32,
      handle,
      name
    };

    init(engine, &importDesc, byteSize);
  }

  void SYCLExternalBuffer::init(SYCLEngine* engine, const void* importDesc, size_t byteSize)
  {
    void* ptr = nullptr;

    ze_device_mem_alloc_desc_t allocDesc{};
    allocDesc.stype = ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC;
    allocDesc.pNext = importDesc;

    auto result = zeMemAllocDevice(static_cast<SYCLDevice*>(engine->getDevice())->getZeContext(),
                                   &allocDesc,
                                   byteSize,
                                   0,
                                   engine->getZeDevice(),
                                   &ptr);

    if (result != ZE_RESULT_SUCCESS)
        throw Exception(Error::InvalidOperation, "failed to import external memory");

    this->ptr      = (char*)ptr;
    this->byteSize = byteSize;
    this->shared   = true;
    this->storage  = Storage::Device;
  }

  SYCLExternalBuffer::~SYCLExternalBuffer()
  {
    zeMemFree(static_cast<SYCLDevice*>(getDevice())->getZeContext(), ptr);
  }

OIDN_NAMESPACE_END