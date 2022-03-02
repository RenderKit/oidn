// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "device.h"
#include "buffer.h"
#include "tensor_accessor.h"
#include <vector>
#include <iostream>
#include <fstream>

namespace oidn {

  // Tensor dimensions
  // Canonical order: CHW / OIHW
  using TensorDims = std::vector<int64_t>;

  std::ostream& operator <<(std::ostream& sm, const TensorDims& dims);

  // Tensor descriptor
  struct TensorDesc
  {
    TensorDims   dims;
    TensorLayout layout;
    DataType     dataType;

    TensorDesc() = default;

    TensorDesc(TensorDims dims, TensorLayout layout, DataType dataType)
      : dims(dims), layout(layout), dataType(dataType) {}

    // Returns the number of dimensions
    OIDN_INLINE int getRank() const { return int(dims.size()); }

    // Returns the number of elements in a 1D tensor
    OIDN_INLINE int getX() const
    {
      assert(dims.size() == 1);
      return int(dims[0]);
    }

    // Returns the number of output channels in the tensor
    OIDN_INLINE int getO() const
    {
      assert(dims.size() >= 4);
      return int(dims[dims.size()-4]);
    }

    // Returns the number of input channels in the tensor
    OIDN_INLINE int getI() const
    {
      assert(dims.size() >= 3);
      return int(dims[dims.size()-3]);
    }

    // Returns the number of channels in the tensor
    OIDN_INLINE int getC() const
    {
      assert(dims.size() >= 3);
      return int(dims[dims.size()-3]);
    }

    // Returns the number of channel blocks in the tensor
    OIDN_INLINE int getCB() const
    {
      return getC() / getBlockSize();
    }

    // Returns the height of the tensor
    OIDN_INLINE int getH() const
    {
      assert(dims.size() >= 2);
      return int(dims[dims.size()-2]);
    }

    // Returns the width of the tensor
    OIDN_INLINE int getW() const
    {
      assert(dims.size() >= 2);
      return int(dims[dims.size()-1]);
    }

    // Returns the block size of the layout
    OIDN_INLINE int getBlockSize() const
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

    // Returns the number of elements in the tensor
    OIDN_INLINE size_t getNumElements() const
    {
      if (dims.empty())
        return 0;

      size_t num = 1;
      for (size_t i = 0; i < dims.size(); ++i)
        num *= dims[i];
      return num;
    }

    // Returns the size in bytes of the tensor
    OIDN_INLINE size_t getByteSize() const
    {
      return getNumElements() * getDataTypeSize(dataType);
    }

    // Returns the aligned size in bytes of the tensor
    OIDN_INLINE size_t getAlignedSize() const
    {
      return round_up(getByteSize(), memoryAlignment);
    }

    bool operator ==(const TensorDesc& other) const
    {
      return (dims == other.dims) && (layout == other.layout) && (dataType == other.dataType);
    }

    bool operator !=(const TensorDesc& other) const
    {
      return (dims != other.dims) || (layout != other.layout) || (dataType != other.dataType);
    }
  };

  // Tensor
  class Tensor : public Memory, protected TensorDesc
  {
  public:
    virtual void* getData() = 0;
    virtual const void* getData() const = 0;

    OIDN_INLINE const TensorDesc& getDesc() const { return *this; }
    OIDN_INLINE const TensorDims& getDims() const { return dims; }
    OIDN_INLINE TensorLayout getLayout() const { return layout; }
    OIDN_INLINE DataType getDataType() const { return dataType; }

    using TensorDesc::getRank;
    using TensorDesc::getX;
    using TensorDesc::getO;
    using TensorDesc::getI;
    using TensorDesc::getC;
    using TensorDesc::getCB;
    using TensorDesc::getH;
    using TensorDesc::getW;
    using TensorDesc::getBlockSize;
    using TensorDesc::getNumElements;
    using TensorDesc::getByteSize;
    using TensorDesc::getAlignedSize;

    OIDN_INLINE operator bool() const { return getData() != nullptr; }

    template<typename T>
    operator TensorAccessor1D<T>() const
    {
      if (layout != TensorLayout::x || dataType != DataTypeOf<T>::value)
        throw Exception(Error::Unknown, "incompatible tensor accessor");
      return TensorAccessor1D<T>(getData(), dims[0]);
    }

    template<typename T, TensorLayout layout>
    operator TensorAccessor3D<T, layout>() const
    {
      if (this->layout != layout || dataType != DataTypeOf<T>::value)
        throw Exception(Error::Unknown, "incompatible tensor accessor");
      return TensorAccessor3D<T, layout>(getData(), getC(), getH(), getW());
    }

    template<typename T, TensorLayout layout>
    operator TensorAccessor4D<T, layout>() const
    {
      if (this->layout != layout || dataType != DataTypeOf<T>::value)
        throw Exception(Error::Unknown, "incompatible tensor accessor");
      return TensorAccessor4D<T, layout>(getData(), getO(), getI(), getH(), getW());
    }

    operator ispc::TensorAccessor3D() const;

    void dump(const std::string& filenamePrefix) const;

  protected:
    Tensor(const Ref<Device>& device, const TensorDesc& desc);
    Tensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset);

    Ref<Device> device;
  };

  class GenericTensor final : public Tensor
  {
  public:
    GenericTensor(const Ref<Device>& device, const TensorDesc& desc);
    GenericTensor(const Ref<Device>& device, const TensorDesc& desc, void* data);
    GenericTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset);

    void* getData() override { return ptr; }
    const void* getData() const override { return ptr; }

  private:
    void init(const Ref<Device>& device);
    void init(const Ref<Device>& device, void* data);
    void updatePtr() override;

    void* ptr;
  };

  // Reorder with zero padding
  void reorder(const Tensor& src, Tensor& dst);

} // namespace oidn
