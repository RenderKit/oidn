// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor.h"

namespace oidn {

  // Returns the number of elements in the tensor
  size_t getNumElements(const TensorDims& dims)
  {
    if (dims.empty())
      return 0;

    size_t num = 1;
    for (size_t i = 0; i < dims.size(); ++i)
      num *= dims[i];
    return num;
  }

  // Returns the maximum tensor dimensions from a list
  TensorDims getMaxDims(const std::vector<TensorDims>& dims)
  {
    TensorDims result;
    size_t maxSize = 0;

    for (const TensorDims& d : dims)
    {
      const size_t size = getNumElements(d);
      if (size > maxSize)
      {
        result = d;
        maxSize = size;
      }
    }

    return result;
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
    if (ndims() != 3 || dataType != DataType::Float32)
      throw Exception(Error::Unknown, "incompatible tensor accessor");

    ispc::TensorAccessor3D result;
    result.ptr = (float*)data();
    result.C = dims[0];
    result.H = dims[1];
    result.W = dims[2];
    return result;
  }

  void Tensor::dump(const std::string& filenamePrefix) const
  {
    assert(ndims() == 3);
    assert(dataType == DataType::Float32);

    const int C = dims[0];
    const int H = dims[1];
    const int W = dims[2];
    const int B = blockSize();

    const float* ptr = (const float*)data();

    for (int c = 0; c < C; ++c)
    {
      // Open the file
      const std::string filename = filenamePrefix + toString(c) + ".pfm";
      std::ofstream file(filename, std::ios::binary);
      if (file.fail())
        throw std::runtime_error("cannot open image file: " + std::string(filename));

      // Write the header
      file << "Pf" << std::endl;
      file << W << " " << H << std::endl;
      file << "-1.0" << std::endl;

      // Write the pixels
      for (int h = H-1; h >= 0; --h)
      {
        for (int w = 0; w < W; ++w)
        {
          const float x = ptr[((size_t)H * (c/B) + h) * ((size_t)W*B) + (size_t)w*B + (c%B)];
          file.write((char*)&x, sizeof(float));
        }
      }
    }
  }

  GenericTensor::GenericTensor(const Ref<Device>& device, const TensorDesc& desc)
    : Tensor(device, desc)
  {
    init(device);
  }

  GenericTensor::GenericTensor(const Ref<Device>& device, const TensorDesc& desc, void* data)
    : Tensor(device, desc)
  {
    init(device, data);
  }

  GenericTensor::GenericTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
    : Tensor(buffer, desc, byteOffset)
  {
    if (byteOffset + byteSize() > buffer->size())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    init(device, buffer->data() + byteOffset);
  }

  void GenericTensor::init(const Ref<Device>& device)
  {
    buffer = device->newBuffer(byteSize(), Buffer::Kind::Device);
    ptr = buffer->data();
  }

  void GenericTensor::init(const Ref<Device>& device, void* data)
  {
    ptr = data;
  }

  void GenericTensor::updatePtr()
  {
    if (buffer)
    {
      if (bufferOffset + byteSize() > buffer->size())
        throw Exception(Error::Unknown, "buffer region out of range");

      ptr = buffer->data() + bufferOffset;
    }
  }

} // namespace oidn
