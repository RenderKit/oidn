// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_tensor.h"

OIDN_NAMESPACE_BEGIN

  DNNLTensor::DNNLTensor(const Ref<DNNLEngine>& engine, const TensorDesc& desc, Storage storage)
    : Tensor(engine->newBuffer(desc.getByteSize(), storage), desc)
  {
    mem = dnnl::memory(toDNNL(getDesc()), engine->getDNNLEngine(), buffer->getPtr());
  }

  DNNLTensor::DNNLTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Tensor(buffer, desc, byteOffset)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    mem = dnnl::memory(toDNNL(getDesc()),
                       static_cast<DNNLEngine*>(buffer->getEngine())->getDNNLEngine(),
                       buffer->getPtr() + byteOffset);
  }

  void DNNLTensor::updatePtr()
  {
    if (buffer)
    {
      if (byteOffset + getByteSize() > buffer->getByteSize())
        throw std::range_error("buffer region out of range");

      mem.set_data_handle(buffer->getPtr() + byteOffset);
    }
  }

OIDN_NAMESPACE_END
