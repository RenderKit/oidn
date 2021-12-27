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
  size_t getNumElements(const TensorDims& dims);

  // Returns the maximum tensor dimensions from a list
  TensorDims getMaxDims(const std::vector<TensorDims>& dims);

  // Tensor descriptor
  struct TensorDesc
  {
    TensorDims   dims;
    TensorLayout layout;
    DataType     dataType;

    OIDN_INLINE TensorDesc() = default;

    OIDN_INLINE TensorDesc(TensorDims dims, TensorLayout layout, DataType dataType)
      : dims(dims), layout(layout), dataType(dataType) {}

    // Returns the number of dimensions
    OIDN_INLINE int ndims() const { return int(dims.size()); }

    // Returns the number of elements in the tensor
    OIDN_INLINE size_t numElements() const
    {
      return getNumElements(dims);
    }

    // Returns the size in bytes of an element in the tensor
    OIDN_INLINE size_t elementByteSize() const
    {
      return getByteSize(dataType);
    }

    // Returns the size in bytes of the tensor
    OIDN_INLINE size_t byteSize() const
    {
      return numElements() * elementByteSize();
    }

    // Returns the aligned size in bytes of the tensor
    OIDN_INLINE size_t alignedByteSize() const
    {
      return round_up(byteSize(), memoryAlignment);
    }

    // Returns the block size of the layout
    OIDN_INLINE int blockSize() const
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
    OIDN_INLINE int numChannels() const
    {
      assert(dims.size() >= 3);
      return int(dims[dims.size()-3]);
    }

    // Returns the number of channel blocks in the tensor
    OIDN_INLINE int numChannelBlocks() const
    {
      return numChannels() / blockSize();
    }

    // Returns the height of the tensor
    OIDN_INLINE int height() const
    {
      assert(dims.size() >= 2);
      return int(dims[dims.size()-2]);
    }

    // Returns the width of the tensor
    OIDN_INLINE int width() const
    {
      assert(dims.size() >= 2);
      return int(dims[dims.size()-1]);
    }

    OIDN_INLINE bool operator ==(const TensorDesc& other) const
    {
      return (dims == other.dims) && (layout == other.layout) && (dataType == other.dataType);
    }

    OIDN_INLINE bool operator !=(const TensorDesc& other) const
    {
      return (dims != other.dims) || (layout != other.layout) || (dataType != other.dataType);
    }
  };

  // Tensor
  class Tensor : public Memory, public TensorDesc
  {
  protected:
    Ref<Device> device;
    
    Tensor(const Ref<Device>& device, const TensorDesc& desc);
    Tensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset);

  public:
    OIDN_INLINE operator bool() const { return data() != nullptr; }

    virtual void* data() = 0;
    virtual const void* data() const = 0;

    OIDN_INLINE const TensorDesc& desc() const { return *this; }

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

    operator ispc::TensorAccessor3D() const;

    void dump(const std::string& filenamePrefix) const;
  };

  class GenericTensor : public Tensor
  {
  private:
    void* ptr;

  public:
    GenericTensor(const Ref<Device>& device, const TensorDesc& desc);
    GenericTensor(const Ref<Device>& device, const TensorDesc& desc, void* data);
    GenericTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset);

    void* data() override { return ptr; }
    const void* data() const override { return ptr; }

  private:
    void init(const Ref<Device>& device);
    void init(const Ref<Device>& device, void* data);
    void updatePtr() override;
  };

} // namespace oidn
