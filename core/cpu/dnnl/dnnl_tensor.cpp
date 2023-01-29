// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_tensor.h"

namespace oidn {

  DNNLTensor::DNNLTensor(const Ref<DNNLEngine>& engine, const TensorDesc& desc, Storage storage)
    : Tensor(engine->newBuffer(desc.getByteSize(), storage), desc)
  {
    init(engine, buffer->getData());
  }

  DNNLTensor::DNNLTensor(const Ref<DNNLEngine>& engine, const TensorDesc& desc, void* data)
    : Tensor(engine, desc)
  {
    init(engine, data);
  }

  DNNLTensor::DNNLTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Tensor(buffer, desc, byteOffset)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    init(staticRefCast<DNNLEngine>(engine), buffer->getData() + byteOffset);
  }

  void DNNLTensor::init(const Ref<DNNLEngine>& engine, void* data)
  {
    mem = dnnl::memory(toDNNL(getDesc()), engine->getDNNLEngine(), data);
  }

  void DNNLTensor::updatePtr()
  {
    if (buffer)
    {
      if (byteOffset + getByteSize() > buffer->getByteSize())
        throw std::range_error("buffer region out of range");

      mem.set_data_handle(buffer->getData() + byteOffset);
    }
  }

} // namespace oidn
