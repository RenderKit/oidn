// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
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
    default:                assert(0);
    }
    return 0;
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
    oihw,
    chw,
    Chw8c,  // blocked
    Chw16c, // blocked
  };

  // Tensor descriptor
  struct TensorDesc
  {
    TensorDims   dims;
    TensorLayout layout;
    DataType     dataType;

    __forceinline TensorDesc() {}

    __forceinline TensorDesc(TensorDims dims, TensorLayout layout, DataType dataType)
      : dims(dims), layout(layout), dataType(dataType) {}

    // Returns the number of dimensions
    __forceinline int ndims() const { return int(dims.size()); }

    // Returns the number of elements in the tensor
    __forceinline size_t numElements() const
    {
      return getNumElements(dims);
    }

    // Return the size in bytes of an element in the tensor
    __forceinline size_t elementByteSize() const
    {
      return getByteSize(dataType);
    }

    // Returns the size in bytes of the tensor
    __forceinline size_t byteSize() const
    {
      return numElements() * elementByteSize();
    }
    
    __forceinline operator dnnl::memory::desc() const
    {
      dnnl::memory::dims memDims;
      dnnl::memory::format_tag memFormat;
      switch (layout)
      {
      case TensorLayout::x:
        assert(ndims() == 1);
        memDims   = {dims[0]};
        memFormat = dnnl::memory::format_tag::x;
        break;
      case TensorLayout::oihw:
        assert(ndims() == 4);
        memDims   = {dims[0], dims[1], dims[2], dims[3]};
        memFormat = dnnl::memory::format_tag::oihw;
        break;
      case TensorLayout::chw:
        assert(ndims() == 3);
        memDims   = {1, dims[0], dims[1], dims[2]};
        memFormat = dnnl::memory::format_tag::nchw;
        break;
      case TensorLayout::Chw8c:
        assert(ndims() == 3);
        memDims   = {1, dims[0], dims[1], dims[2]};
        memFormat = dnnl::memory::format_tag::nChw8c;
        break;
      case TensorLayout::Chw16c:
        assert(ndims() == 3);
        memDims   = {1, dims[0], dims[1], dims[2]};
        memFormat = dnnl::memory::format_tag::nChw16c;
        break;
      default:
        assert(0);
      }

      dnnl::memory::data_type memType;
      switch (dataType)
      {
      case DataType::Float32:
        memType = dnnl::memory::data_type::f32;
        break;
      case DataType::UInt8:
        memType = dnnl::memory::data_type::u8;
        break;
      default:
        assert(0);
      }

      return dnnl::memory::desc(memDims, memType, memFormat);
    }
  };

  // Tensor
  class Tensor : public RefCount, public TensorDesc
  {
  public:
    dnnl::memory mem;

  private:
    Ref<Device> device;
    Ref<Tensor> parent; // for views

  public:
    __forceinline Tensor() {}

    __forceinline Tensor(const Ref<Device>& device, const TensorDesc& desc)
      : TensorDesc(desc), device(device)
    {
      init(device);
    }

    __forceinline Tensor(const Ref<Device>& device, const TensorDesc& desc, void* data)
      : TensorDesc(desc), device(device)
    {
      init(device, data);
    }

    __forceinline Tensor(const Ref<Device>& device, TensorDims dims, TensorLayout layout, DataType dataType)
      : TensorDesc(dims, layout, dataType), device(device)
    {
      init(device);
    }

    __forceinline Tensor(const Ref<Device>& device, TensorDims dims, TensorLayout layout, DataType dataType, void* data)
      : TensorDesc(dims, layout, dataType), device(device)
    {
      init(device, data);
    }

    __forceinline Tensor(const Ref<Device>& device, const dnnl::memory::desc& desc)
      : TensorDesc({int64_t(desc.get_size())}, TensorLayout::x, DataType::UInt8),
        mem(desc, device->getEngine()),
        device(device)
    {
    }

    __forceinline operator bool() const { return data() != nullptr; }

    __forceinline void* data() { return mem.get_data_handle(); }
    __forceinline const void* data() const { return mem.get_data_handle(); }

    // Returns a view of the tensor, optionally applying an offset in number of elements
    __forceinline Ref<Tensor> view(const TensorDesc& newDesc, size_t offset = 0)
    {
      size_t byteOffset = offset * newDesc.elementByteSize();
      assert(byteSize() >= newDesc.byteSize() + byteOffset);
      void* newData = (char*)data() + byteOffset;
      Ref<Tensor> result = makeRef<Tensor>(device, newDesc, newData);
      result->parent = this;
      return result;
    }

    __forceinline Ref<Tensor> view(const TensorDims& newDims, size_t offset = 0)
    {
      TensorDesc newDesc {newDims, layout, dataType};
      return view(newDesc, offset);
    }

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

  private:
    void init(const Ref<Device>& device)
    {
      mem = dnnl::memory(*this, device->getEngine());
    }

    void init(const Ref<Device>& device, void* data)
    {
      mem = dnnl::memory(*this, device->getEngine(), data);
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
  };

} // namespace oidn
