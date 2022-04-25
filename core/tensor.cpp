// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor.h"

namespace oidn {

  std::ostream& operator <<(std::ostream& sm, const TensorDims& dims)
  {
    sm << "[";
    for (size_t i = 0; i < dims.size(); ++i)
    {
      if (i > 0)
        sm << ", ";
      sm << dims[i];
    }
    sm << "]";
    return sm;
  }

  Tensor::Tensor(const Ref<Device>& device, const TensorDesc& desc)
    : TensorDesc(desc),
      device(device) {}

  Tensor::Tensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Memory(buffer, byteOffset),
      TensorDesc(desc),
      device(buffer->getDevice()) {}

  Tensor::operator ispc::TensorAccessor3D() const
  {
    if (getRank() != 3 || dataType != DataType::Float32)
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor3D result;
    result.ptr = (float*)getData();
    result.C = getC();
    result.H = getH();
    result.W = getW();
    return result;
  }

  std::shared_ptr<Tensor> Tensor::map(Access access)
  {
    if (!buffer)
      throw std::logic_error("tensor not backed by a buffer cannot be mapped");
    return device->newTensor(makeRef<MappedBuffer>(buffer, byteOffset, getByteSize(), access), getDesc());
  }

  void Tensor::dump(const std::string& filenamePrefix) const
  {
    if (dataType == DataType::Float32 && layout == TensorLayout::chw)
      dumpImpl<float, TensorLayout::chw>(filenamePrefix);
    else if (dataType == DataType::Float32 && layout == TensorLayout::Chw8c)
      dumpImpl<float, TensorLayout::Chw8c>(filenamePrefix);
    else if (dataType == DataType::Float32 && layout == TensorLayout::Chw16c)
      dumpImpl<float, TensorLayout::Chw16c>(filenamePrefix);
    else if (dataType == DataType::Float16 && layout == TensorLayout::chw)
      dumpImpl<half, TensorLayout::chw>(filenamePrefix);
    else if (dataType == DataType::Float16 && layout == TensorLayout::Chw16c)
      dumpImpl<half, TensorLayout::Chw16c>(filenamePrefix);
    else if (dataType == DataType::Float16 && layout == TensorLayout::hwc)
      dumpImpl<half, TensorLayout::hwc>(filenamePrefix);
    else
      throw std::runtime_error("tensor dump not implemented");
  }

  template<typename T, TensorLayout layout>
  void Tensor::dumpImpl(const std::string& filenamePrefix) const
  {
    TensorAccessor3D<T, layout> acc = *this;

    for (int c = 0; c < acc.C; ++c)
    {
      // Open the file
      const std::string filename = filenamePrefix + toString(c) + ".pfm";
      std::ofstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open image file: " + std::string(filename));

      // Write the header
      file << "Pf" << std::endl;
      file << acc.W << " " << acc.H << std::endl;
      file << "-1.0" << std::endl;

      // Write the pixels
      for (int h = acc.H-1; h >= 0; --h)
      {
        for (int w = 0; w < acc.W; ++w)
        {
          const float x = acc(c, h, w);
          file.write((char*)&x, sizeof(float));
        }
      }
    }
  }

  GenericTensor::GenericTensor(const Ref<Device>& device, const TensorDesc& desc, Storage storage)
    : Tensor(device, desc)
  {
    init(device, storage);
  }

  GenericTensor::GenericTensor(const Ref<Device>& device, const TensorDesc& desc, void* data)
    : Tensor(device, desc)
  {
    init(device, data);
  }

  GenericTensor::GenericTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Tensor(buffer, desc, byteOffset)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    init(device, buffer->getData() + byteOffset);
  }

  void GenericTensor::init(const Ref<Device>& device, Storage storage)
  {
    buffer = device->newBuffer(getByteSize(), storage);
    ptr = buffer->getData();
  }

  void GenericTensor::init(const Ref<Device>& device, void* data)
  {
    ptr = data;
  }

  void GenericTensor::updatePtr()
  {
    if (buffer)
    {
      if (byteOffset + getByteSize() > buffer->getByteSize())
        throw std::range_error("buffer region out of range");

      ptr = buffer->getData() + byteOffset;
    }
  }

} // namespace oidn
