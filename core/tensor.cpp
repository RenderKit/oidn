// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tensor.h"
#include "tensor_accessor.h"

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
      throw Exception(Error::Unknown, "incompatible tensor accessor");

    ispc::TensorAccessor3D result;
    result.ptr = (float*)getData();
    result.C = getC();
    result.H = getH();
    result.W = getW();
    return result;
  }

  void Tensor::dump(const std::string& filenamePrefix) const
  {
    assert(getRank() == 3);
    assert(dataType == DataType::Float32);

    const int C = getC();
    const int H = getH();
    const int W = getW();
    const int B = getBlockSize();

    const float* ptr = (const float*)getData();

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
    if (byteOffset + getByteSize() > buffer->getByteSize())
      throw Exception(Error::InvalidArgument, "buffer region out of range");

    init(device, buffer->getData() + byteOffset);
  }

  void GenericTensor::init(const Ref<Device>& device)
  {
    buffer = device->newBuffer(getByteSize(), MemoryKind::Shared);
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
      if (bufferOffset + getByteSize() > buffer->getByteSize())
        throw Exception(Error::Unknown, "buffer region out of range");

      ptr = buffer->getData() + bufferOffset;
    }
  }

  namespace
  {
    template<typename SrcT, typename DstT, TensorLayout srcLayout, TensorLayout dstLayout>
    struct Reorder
    {
      void operator ()(const Tensor& src, Tensor& dst)
      {
        TensorAccessor4D<SrcT, srcLayout> srcAcc = src;
        TensorAccessor4D<DstT, dstLayout> dstAcc = dst;

        for (int o = 0; o < dstAcc.O; ++o)
        {
          for (int i = 0; i < dstAcc.I; ++i)
          {
            for (int h = 0; h < dstAcc.H; ++h)
            {
              for (int w = 0; w < dstAcc.W; ++w)
              {
                SrcT value;
                if (o < srcAcc.O && i < srcAcc.I)
                  value = srcAcc(o, i, h, w);
                else
                  value = 0; // padding

                dstAcc(o, i, h, w) = DstT(value);
              }
            }
          }
        }
      }
    };

    template<typename SrcT, typename DstT>
    struct Reorder<SrcT, DstT, TensorLayout::x, TensorLayout::x>
    {
      void operator ()(const Tensor& src, Tensor& dst)
      {
        TensorAccessor1D<SrcT> srcAcc = src;
        TensorAccessor1D<DstT> dstAcc = dst;

        for (int x = 0; x < srcAcc.X; ++x)
          dstAcc(x) = srcAcc(x);

        for (int x = srcAcc.X; x < dstAcc.X; ++x)
          dstAcc(x) = 0; // padding
      }
    };

    template<typename SrcT, typename DstT, TensorLayout srcLayout, TensorLayout dstLayout>
    bool tryReorder(const Tensor& src, Tensor& dst)
    {
      if (src.getDataType() == DataTypeOf<SrcT>::value && src.getLayout() == srcLayout &&
          dst.getDataType() == DataTypeOf<DstT>::value && dst.getLayout() == dstLayout)
      {
        Reorder<SrcT, DstT, srcLayout, dstLayout>()(src, dst);
        return true;
      }

      return false;
    }
  }

  void reorder(const Tensor& src, Tensor& dst)
  {
    bool ok =
      tryReorder<float, float, TensorLayout::x,    TensorLayout::x>(src, dst) ||
      tryReorder<float, half,  TensorLayout::x,    TensorLayout::x>(src, dst) ||
      tryReorder<float, float, TensorLayout::oihw, TensorLayout::oihw>(src, dst) ||
      tryReorder<float, half,  TensorLayout::oihw, TensorLayout::oihw>(src, dst) ||
      tryReorder<float, float, TensorLayout::oihw, TensorLayout::ohwi>(src, dst) ||
      tryReorder<float, half,  TensorLayout::oihw, TensorLayout::ohwi>(src, dst);

    if (!ok)
      throw Exception(Error::Unknown, "unsupported reorder");
  }

} // namespace oidn
