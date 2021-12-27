// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_tensor.h"

namespace oidn {

  DNNLTensor::DNNLTensor(const Ref<DNNLDevice>& device, const TensorDesc& desc)
    : Tensor(device, desc)
  {
    init(device);
  }

  DNNLTensor::DNNLTensor(const Ref<DNNLDevice>& device, const TensorDesc& desc, void* data)
    : Tensor(device, desc)
  {
    init(device, data);
  }

  DNNLTensor::DNNLTensor(const Ref<DNNLDevice>& device, const dnnl::memory::desc& desc)
    : Tensor(device, TensorDesc({int64_t(desc.get_size())}, TensorLayout::x, DataType::UInt8)),
      mem(desc, device->getDNNLEngine()) {}

  DNNLTensor::DNNLTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Tensor(buffer, desc, byteOffset)
  {
    if (byteOffset + byteSize() > buffer->size())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    init(dynamicRefCast<DNNLDevice>(device), buffer->data() + byteOffset);
  }

  void DNNLTensor::init(const Ref<DNNLDevice>& device)
  {
    mem = dnnl::memory(toMemoryDesc(*this), device->getDNNLEngine());
  }

  void DNNLTensor::init(const Ref<DNNLDevice>& device, void* data)
  {
    mem = dnnl::memory(toMemoryDesc(*this), device->getDNNLEngine(), data);
  }

  void DNNLTensor::updatePtr()
  {
    if (buffer)
    {
      if (bufferOffset + byteSize() > buffer->size())
        throw Exception(Error::Unknown, "buffer region out of range");

      mem.set_data_handle(buffer->data() + bufferOffset);
    }
  }

  const dnnl::memory& DNNLTensor::getMemory(const Tensor& tz)
  {
    if (auto dnnlTensor = dynamic_cast<const DNNLTensor*>(&tz))
      return dnnlTensor->mem;
    else
      throw Exception(Error::Unknown, "not DNNL tensor");
  }

  dnnl::memory::desc DNNLTensor::toMemoryDesc(const TensorDesc& tz)
  {
    dnnl::memory::dims dnnlDims;
    dnnl::memory::format_tag dnnlFormat;
    switch (tz.layout)
    {
    case TensorLayout::x:
      assert(ndims() == 1);
      dnnlDims   = {tz.dims[0]};
      dnnlFormat = dnnl::memory::format_tag::x;
      break;
    case TensorLayout::chw:
      assert(ndims() == 3);
      dnnlDims   = {1, tz.dims[0], tz.dims[1], tz.dims[2]};
      dnnlFormat = dnnl::memory::format_tag::nchw;
      break;
    case TensorLayout::Chw8c:
      assert(ndims() == 3);
      dnnlDims   = {1, tz.dims[0], tz.dims[1], tz.dims[2]};
      dnnlFormat = dnnl::memory::format_tag::nChw8c;
      break;
    case TensorLayout::Chw16c:
      assert(ndims() == 3);
      dnnlDims   = {1, tz.dims[0], tz.dims[1], tz.dims[2]};
      dnnlFormat = dnnl::memory::format_tag::nChw16c;
      break;
    case TensorLayout::oihw:
      assert(ndims() == 4);
      dnnlDims   = {tz.dims[0], tz.dims[1], tz.dims[2], tz.dims[3]};
      dnnlFormat = dnnl::memory::format_tag::oihw;
      break;
    default:
      throw Exception(Error::Unknown, "invalid tensor layout");
    }

    dnnl::memory::data_type dnnlType;
    switch (tz.dataType)
    {
    case DataType::Float32:
      dnnlType = dnnl::memory::data_type::f32;
      break;
    case DataType::Float16:
      dnnlType = dnnl::memory::data_type::f16;
      break;
    case DataType::UInt8:
      dnnlType = dnnl::memory::data_type::u8;
      break;
    default:
      throw Exception(Error::Unknown, "invalid tensor data type");
    }

    return dnnl::memory::desc(dnnlDims, dnnlType, dnnlFormat);
  }

} // namespace oidn
