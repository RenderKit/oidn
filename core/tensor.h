// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "buffer.h"
#include <vector>

namespace oidn {
  
  // Tensor data type
  enum class DataType
  {
    Float32,
    UInt8,
  };

  // Returns the size of the specified data type in bytes
  __forceinline size_t getByteSize(DataType dataType)
  {
    switch (dataType)
    {
    case DataType::Float32: return 4;
    case DataType::UInt8:   return 1;
    default:
      throw Exception(Error::Unknown, "invalid tensor data type");
    }
  }

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

  // Tensor memory layout
  enum class TensorLayout
  {
    x,
    chw,
    Chw8c,  // blocked
    Chw16c, // blocked
    oihw,
  };

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

    template <typename T> __forceinline T& get(int64_t i0)
    { return ((T*)data())[getIndex(i0)]; }
    template <typename T> __forceinline const T& get(int64_t i0) const
    { return ((T*)data())[getIndex(i0)]; }

    template <typename T> __forceinline T& get(int64_t i0, int64_t i1, int64_t i2)
    { return ((T*)data())[getIndex(i0, i1, i2)]; }
    template <typename T> __forceinline const T& get(int64_t i0, int64_t i1, int64_t i2) const
    { return ((T*)data())[getIndex(i0, i1, i2)]; }

    template <typename T> __forceinline T& get(int64_t i0, int64_t i1, int64_t i2, int64_t i3)
    { return ((T*)data())[getIndex(i0, i1, i2, i3)]; }
    template <typename T> __forceinline const T& get(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const
    { return ((T*)data())[getIndex(i0, i1, i2, i3)]; }

    // Converts to ISPC equivalent
    operator ispc::Tensor() const
    {
      assert(ndims() == 3);
      assert(dataType == DataType::Float32);

      ispc::Tensor result;
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
      buffer = device->newBuffer(byteSize());
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

    __forceinline int64_t getIndex(int64_t i0) const
    {
      assert(ndims() == 1);
      assert(i0 < dims[0]);
      return i0;
    }

    __forceinline int64_t getIndex(int64_t i0, int64_t i1, int64_t i2) const
    {
      assert(ndims() == 3);
      assert(layout == TensorLayout::chw);
      assert(i0 < dims[0] && i1 < dims[1] && i2 < dims[2]);
      return (i0 * dims[1] + i1) * dims[2] + i2;
    }

    __forceinline int64_t getIndex(int64_t i0, int64_t i1, int64_t i2, int64_t i3) const
    {
      assert(ndims() == 4);
      assert(layout == TensorLayout::oihw);
      assert(i0 < dims[0] && i1 < dims[1] && i2 < dims[2] && i3 < dims[3]);
      return ((i0 * dims[1] + i1) * dims[2] + i2) * dims[3] + i3;
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
  };

} // namespace oidn
