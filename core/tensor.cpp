// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor.h"
#include "engine.h"
#include <fstream>

OIDN_NAMESPACE_BEGIN

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

  Tensor::Tensor(const TensorDesc& desc)
    : TensorDesc(desc)
  {
    assert(desc.isValid());
  }

  Tensor::Tensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Memory(buffer, byteOffset),
      TensorDesc(desc)
  {
    assert(desc.isValid());
  }

  std::shared_ptr<Tensor> Tensor::map(Access access)
  {
    if (!buffer)
      throw std::logic_error("tensor not backed by a buffer cannot be mapped");
    return buffer->getEngine()->newTensor(makeRef<MappedBuffer>(buffer, byteOffset, getByteSize(), access), getDesc());
  }

  void Tensor::dump(const std::string& filenamePrefix)
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

  template<typename T, TensorLayout accessorLayout>
  void Tensor::dumpImpl(const std::string& filenamePrefix)
  {
    TensorAccessor3D<T, accessorLayout> acc = *this;

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

  GenericTensor::GenericTensor(const TensorDesc& desc)
    : Tensor(desc),
      ptr(alignedMalloc(getByteSize())),
      shared(false) {}

  GenericTensor::GenericTensor(const TensorDesc& desc, void* data)
    : Tensor(desc),
      ptr(data),
      shared(true) {}

  GenericTensor::GenericTensor(const Ref<Engine>& engine, const TensorDesc& desc, Storage storage)
    : Tensor(desc),
      shared(true)
  {
    buffer = engine->newBuffer(getByteSize(), storage);
    ptr = buffer->getPtr();
  }

  GenericTensor::GenericTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Tensor(buffer, desc, byteOffset),
      shared(true)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    ptr = static_cast<char*>(buffer->getPtr()) + byteOffset;
  }

  GenericTensor::~GenericTensor()
  {
    if (!shared)
      alignedFree(ptr);
  }

  void GenericTensor::updatePtr()
  {
    if (buffer)
    {
      if (byteOffset + getByteSize() > buffer->getByteSize())
        throw std::range_error("buffer region out of range");

      ptr = static_cast<char*>(buffer->getPtr()) + byteOffset;
    }
  }

OIDN_NAMESPACE_END
