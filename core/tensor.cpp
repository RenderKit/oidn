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

  // -----------------------------------------------------------------------------------------------
  // Tensor
  // -----------------------------------------------------------------------------------------------

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

  // -----------------------------------------------------------------------------------------------
  // HostTensor
  // -----------------------------------------------------------------------------------------------

  HostTensor::HostTensor(const TensorDesc& desc)
    : Tensor(desc),
      ptr(alignedMalloc(getByteSize())),
      shared(false) {}

  HostTensor::HostTensor(const TensorDesc& desc, void* data)
    : Tensor(desc),
      ptr(data),
      shared(true) {}

  HostTensor::~HostTensor()
  {
    if (!shared)
      alignedFree(ptr);
  }

  std::shared_ptr<Tensor> HostTensor::toDevice(const Ref<Engine>& engine, Storage storage)
  {
    const size_t byteSize = getByteSize();
    auto bufferCopy = engine->newBuffer(byteSize, storage);
    bufferCopy->write(0, byteSize, getData());
    return engine->newTensor(bufferCopy, getDesc());
  }

  // -----------------------------------------------------------------------------------------------
  // DeviceTensor
  // -----------------------------------------------------------------------------------------------

  DeviceTensor::DeviceTensor(const Ref<Engine>& engine, const TensorDesc& desc, Storage storage)
    : Tensor(desc)
  {
    buffer = engine->newBuffer(getByteSize(), storage);
    ptr = buffer->getData();
  }

  DeviceTensor::DeviceTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Tensor(buffer, desc, byteOffset)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region is out of range");

    ptr = buffer->getData() + byteOffset;
  }

  void DeviceTensor::updatePtr()
  {
    if (buffer)
    {
      if (byteOffset + getByteSize() > buffer->getByteSize())
        throw std::range_error("buffer region is out of range");

      ptr = buffer->getData() + byteOffset;
    }
  }

OIDN_NAMESPACE_END
