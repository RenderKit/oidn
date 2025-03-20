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

  Ref<Tensor> Tensor::toDevice(Engine* engine, Storage storage)
  {
    return this;
  }

#if 0
  uint32_t Tensor::getHash() const
  {
    std::vector<uint8_t> hostBytes;
    const uint8_t* bytes;

    if (buffer && buffer->getStorage() == Storage::Device)
    {
      // Copy the tensor to the host
      hostBytes.resize(getByteSize());
      getBuffer()->getEngine()->usmCopy(hostBytes.data(), getPtr(), getByteSize());
      bytes = hostBytes.data();
    }
    else
    {
      bytes = static_cast<const uint8_t*>(getPtr());
    }

    const size_t numBytes = getByteSize();
    uint32_t hash = 0x811c9dc5;
    for (size_t i = 0; i < numBytes; ++i)
    {
      hash ^= bytes[i];
      hash *= 0x1000193;
    }
    return hash;
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
    if (buffer && buffer->getStorage() == Storage::Device)
      throw std::runtime_error("tensor dump not implemented for device storage");

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
#endif

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

  Ref<Tensor> HostTensor::toDevice(Engine* engine, Storage storage)
  {
    const size_t byteSize = getByteSize();
    auto bufferCopy = engine->newBuffer(byteSize, storage);
    bufferCopy->write(0, byteSize, getPtr());
    return engine->newTensor(bufferCopy, getDesc());
  }

  // -----------------------------------------------------------------------------------------------
  // DeviceTensor
  // -----------------------------------------------------------------------------------------------

  DeviceTensor::DeviceTensor(Engine* engine, const TensorDesc& desc, Storage storage)
    : Tensor(desc)
  {
    buffer = engine->newBuffer(getByteSize(), storage);
    ptr = buffer->getPtr();
  }

  DeviceTensor::DeviceTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Tensor(buffer, desc, byteOffset)
  {
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region is out of bounds");

    ptr = static_cast<char*>(buffer->getPtr()) + byteOffset;
  }

  void DeviceTensor::postRealloc()
  {
    if (buffer)
      ptr = static_cast<char*>(buffer->getPtr()) + byteOffset;
  }

OIDN_NAMESPACE_END
