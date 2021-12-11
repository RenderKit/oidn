// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "buffer.h"
#include "tensor_accessor.h"
#include <vector>
#include <fstream>

namespace oidn {

  // Tensor dimensions
  using TensorDims = std::vector<int64_t>;

  // Returns the number of elements in the tensor
  __forceinline size_t getNumElements(const TensorDims& dims)
  {
    if (dims.empty())
      return 0;

    size_t num = 1;
    for (size_t i = 0; i < dims.size(); ++i)
      num *= dims[i];
    return num;
  }

  // Returns the maximum tensor dimensions from a list
  inline TensorDims getMaxDims(const std::vector<TensorDims>& dims)
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

  // Tensor descriptor
  struct TensorDesc
  {
    TensorDims   dims;
    TensorLayout layout;
    DataType     dataType;

    __forceinline TensorDesc() = default;

    __forceinline TensorDesc(TensorDims dims, TensorLayout layout, DataType dataType)
      : dims(dims), layout(layout), dataType(dataType) {}

    // Returns the number of dimensions
    __forceinline int ndims() const { return int(dims.size()); }

    // Returns the number of elements in the tensor
    __forceinline size_t numElements() const
    {
      return getNumElements(dims);
    }

    // Returns the size in bytes of an element in the tensor
    __forceinline size_t elementByteSize() const
    {
      return getByteSize(dataType);
    }

    // Returns the size in bytes of the tensor
    __forceinline size_t byteSize() const
    {
      return numElements() * elementByteSize();
    }

    // Returns the aligned size in bytes of the tensor
    __forceinline size_t alignedByteSize() const
    {
      return round_up(byteSize(), memoryAlignment);
    }

    // Returns the block size of the layout
    __forceinline int blockSize() const
    {
      switch (layout)
      {
      case TensorLayout::Chw8c:
        return 8;
      case TensorLayout::Chw16c:
        return 16;
      default:
        return 1;
      }
    }

    // Returns the number of channels in the tensor
    __forceinline int numChannels() const
    {
      assert(dims.size() >= 3);
      return int(dims[dims.size()-3]);
    }

    // Returns the number of channel blocks in the tensor
    __forceinline int numChannelBlocks() const
    {
      return numChannels() / blockSize();
    }

    // Returns the height of the tensor
    __forceinline int height() const
    {
      assert(dims.size() >= 2);
      return int(dims[dims.size()-2]);
    }

    // Returns the width of the tensor
    __forceinline int width() const
    {
      assert(dims.size() >= 2);
      return int(dims[dims.size()-1]);
    }

    __forceinline bool operator ==(const TensorDesc& other) const
    {
      return (dims == other.dims) && (layout == other.layout) && (dataType == other.dataType);
    }

    __forceinline bool operator !=(const TensorDesc& other) const
    {
      return (dims != other.dims) || (layout != other.layout) || (dataType != other.dataType);
    }
    
  #if defined(OIDN_DNNL)
    operator dnnl::memory::desc() const
    {
      dnnl::memory::dims dnnlDims;
      dnnl::memory::format_tag dnnlFormat;
      switch (layout)
      {
      case TensorLayout::x:
        assert(ndims() == 1);
        dnnlDims   = {dims[0]};
        dnnlFormat = dnnl::memory::format_tag::x;
        break;
      case TensorLayout::chw:
        assert(ndims() == 3);
        dnnlDims   = {1, dims[0], dims[1], dims[2]};
        dnnlFormat = dnnl::memory::format_tag::nchw;
        break;
      case TensorLayout::Chw8c:
        assert(ndims() == 3);
        dnnlDims   = {1, dims[0], dims[1], dims[2]};
        dnnlFormat = dnnl::memory::format_tag::nChw8c;
        break;
      case TensorLayout::Chw16c:
        assert(ndims() == 3);
        dnnlDims   = {1, dims[0], dims[1], dims[2]};
        dnnlFormat = dnnl::memory::format_tag::nChw16c;
        break;
      case TensorLayout::oihw:
        assert(ndims() == 4);
        dnnlDims   = {dims[0], dims[1], dims[2], dims[3]};
        dnnlFormat = dnnl::memory::format_tag::oihw;
        break;
      default:
        throw Exception(Error::Unknown, "invalid tensor layout");
      }

      dnnl::memory::data_type dnnlType;
      switch (dataType)
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
  #endif
  };

  // Tensor
  class Tensor : public Memory, public TensorDesc
  {
  public:
  #if defined(OIDN_DNNL)
    dnnl::memory mem;
  #else
    void* ptr;
  #endif
    Ref<Device> device;

  public:
    Tensor(const Ref<Device>& device, const TensorDesc& desc)
      : TensorDesc(desc),
        device(device)
    {
      init(device);
    }

    Tensor(const Ref<Device>& device, const TensorDesc& desc, void* data)
      : TensorDesc(desc),
        device(device)
    {
      init(device, data);
    }

    Tensor(const Ref<Device>& device, TensorDims dims, TensorLayout layout, DataType dataType)
      : TensorDesc(dims, layout, dataType),
        device(device)
    {
      init(device);
    }

    Tensor(const Ref<Device>& device, TensorDims dims, TensorLayout layout, DataType dataType, void* data)
      : TensorDesc(dims, layout, dataType),
        device(device)
    {
      init(device, data);
    }

  #if defined(OIDN_DNNL)
    Tensor(const Ref<Device>& device, const dnnl::memory::desc& desc)
      : TensorDesc({int64_t(desc.get_size())}, TensorLayout::x, DataType::UInt8),
        mem(desc, device->getDNNLEngine()),
        device(device)
    {
    }
  #endif

    Tensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
      : Memory(buffer, byteOffset),
        TensorDesc(desc),
        device(buffer->getDevice())
    {
      if (byteOffset + byteSize() > buffer->size())
        throw Exception(Error::InvalidArgument, "buffer region out of range");

      init(device, buffer->data() + byteOffset);
    }

    __forceinline operator bool() const { return data() != nullptr; }

  #if defined(OIDN_DNNL)
    __forceinline void* data() { return mem.get_data_handle(); }
    __forceinline const void* data() const { return mem.get_data_handle(); }
  #else
    __forceinline void* data() { return ptr; }
    __forceinline const void* data() const { return ptr; }
  #endif

    __forceinline const TensorDesc& desc() const { return *this; }

    template<typename T>
    operator TensorAccessor1D<T>() const
    {
      if (this->layout != TensorLayout::x || this->dataType != DataTypeOf<T>::value)
        throw Exception(Error::Unknown, "incompatible tensor accessor");
      return TensorAccessor1D<T>(data(), dims[0]);
    }

    template<typename T, TensorLayout layout>
    operator TensorAccessor3D<T, layout>() const
    {
      if (this->layout != layout || this->dataType != DataTypeOf<T>::value)
        throw Exception(Error::Unknown, "incompatible tensor accessor");
      return TensorAccessor3D<T, layout>(data(), dims[0], dims[1], dims[2]);
    }

    template<typename T, TensorLayout layout>
    operator TensorAccessor4D<T, layout>() const
    {
      if (this->layout != layout || this->dataType != DataTypeOf<T>::value)
        throw Exception(Error::Unknown, "incompatible tensor accessor");
      return TensorAccessor4D<T, layout>(data(), dims[0], dims[1], dims[2], dims[3]);
    }

    operator ispc::TensorAccessor3D() const
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

  #if defined(OIDN_BNNS)
    operator BNNSNDArrayDescriptor() const
    {
      BNNSNDArrayDescriptor bnnsDesc;
  
      switch (layout)
      {
      case TensorLayout::x:
        assert(ndims() == 1);
        bnnsDesc = BNNSNDArrayDescriptor({
          .layout = BNNSDataLayoutVector,
          .size   = {size_t(dims[0])}
        });
        break;
      case TensorLayout::chw:
        assert(ndims() == 3);
        bnnsDesc = BNNSNDArrayDescriptor({
          .layout = BNNSDataLayoutImageCHW,
          .size   = {size_t(dims[2]), size_t(dims[1]), size_t(dims[0])}
        });
        break;
      case TensorLayout::oihw:
        assert(ndims() == 4);
        bnnsDesc = BNNSNDArrayDescriptor({
          .layout = BNNSDataLayoutConvolutionWeightsOIHW,
          .size   = {size_t(dims[3]), size_t(dims[2]), size_t(dims[1]), size_t(dims[0])}
        });
        break;
      default:
        throw Exception(Error::Unknown, "invalid tensor layout");
      }

      switch (dataType)
      {
      case DataType::Float32:
        bnnsDesc.data_type = BNNSDataTypeFloat32;
        break;
      case DataType::Float16:
        bnnsDesc.data_type = BNNSDataTypeFloat16;
        break;
      case DataType::UInt8:
        bnnsDesc.data_type = BNNSDataTypeUInt8;
        break;
      default:
        throw Exception(Error::Unknown, "invalid tensor data type");
      }

      bnnsDesc.data = ptr;
      return bnnsDesc;
    }
  #endif

  private:
    void init(const Ref<Device>& device)
    {
    #if defined(OIDN_DNNL)
      mem = dnnl::memory(*this, device->getDNNLEngine());
    #else
      buffer = device->newBuffer(byteSize(), Buffer::Kind::Device);
      ptr = buffer->data();
    #endif
    }

    void init(const Ref<Device>& device, void* data)
    {
    #if defined(OIDN_DNNL)
      mem = dnnl::memory(*this, device->getDNNLEngine(), data);
    #else
      ptr = data;
    #endif
    }

    void updatePtr() override
    {
      if (buffer)
      {
        if (bufferOffset + byteSize() > buffer->size())
          throw Exception(Error::Unknown, "buffer region out of range");

      #if defined(OIDN_DNNL)
        mem.set_data_handle(buffer->data() + bufferOffset);
      #else
        ptr = buffer->data() + bufferOffset;
      #endif
      }
    }

  public:
    void dump(const std::string& filenamePrefix) const
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
  };

} // namespace oidn
