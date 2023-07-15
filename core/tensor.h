// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.h"
#include "tensor_accessor.h"
#include <vector>
#include <unordered_map>
#include <iostream>

OIDN_NAMESPACE_BEGIN

  class Engine;

  // Tensor dimensions
  // Canonical order: CHW / OIHW
  using TensorDims = std::vector<int>;

  std::ostream& operator <<(std::ostream& sm, const TensorDims& dims);

  // Tensor descriptor
  struct TensorDesc
  {
    TensorDims   dims;       // logical dimensions
    TensorDims   paddedDims; // storage dimensions with zero-padding
    TensorLayout layout;     // storage layout
    DataType     dataType;   // element data type

    TensorDesc() = default;

    TensorDesc(TensorDims dims, TensorDims paddedDims, TensorLayout layout, DataType dataType)
      : dims(dims), paddedDims(paddedDims), layout(layout), dataType(dataType)
    {
      assert(isValid());
    }

    TensorDesc(TensorDims dims, TensorLayout layout, DataType dataType)
      : dims(dims), paddedDims(dims), layout(layout), dataType(dataType)
    {
      assert(isValid());
    }

    bool isValid() const
    {
      const auto info = getTensorLayoutInfo(layout);

      return getRank() == info.rank &&
             dims.size() == paddedDims.size() &&
             std::mismatch(dims.begin(), dims.end(), paddedDims.begin(),
                           [](int a, int b) { return a <= b; }).first == dims.end() &&
             (info.blockC == 1 ||
               (getRank() == 3 && getPaddedC() % info.blockC == 0) ||
               (getRank() == 4 && getPaddedO() % info.blockC == 0 && getPaddedI() % info.blockC == 0));
    }

    // Returns the number of dimensions
    OIDN_INLINE int getRank() const { return int(dims.size()); }

    // Returns the number of elements in a 1D tensor
    OIDN_INLINE int getX() const
    {
      assert(dims.size() == 1);
      return dims[0];
    }

    OIDN_INLINE int getPaddedX() const
    {
      assert(paddedDims.size() == 1);
      return paddedDims[0];
    }

    // Returns the number of output channels in the tensor
    OIDN_INLINE int getO() const
    {
      assert(dims.size() >= 4);
      return dims[dims.size()-4];
    }

    OIDN_INLINE int getPaddedO() const
    {
      assert(paddedDims.size() >= 4);
      return paddedDims[paddedDims.size()-4];
    }

    // Returns the number of input channels in the tensor
    OIDN_INLINE int getI() const
    {
      assert(dims.size() >= 3);
      return dims[dims.size()-3];
    }

    OIDN_INLINE int getPaddedI() const
    {
      assert(paddedDims.size() >= 3);
      return paddedDims[paddedDims.size()-3];
    }

    // Returns the number of channels in the tensor
    OIDN_INLINE int getC() const
    {
      assert(dims.size() >= 3);
      return dims[dims.size()-3];
    }

    OIDN_INLINE int getPaddedC() const
    {
      assert(paddedDims.size() >= 3);
      return paddedDims[paddedDims.size()-3];
    }

    // Returns the height of the tensor
    OIDN_INLINE int getH() const
    {
      assert(dims.size() >= 2);
      return dims[dims.size()-2];
    }

    // Returns the width of the tensor
    OIDN_INLINE int getW() const
    {
      assert(dims.size() >= 2);
      return dims[dims.size()-1];
    }

    // Returns the number of elements in the tensor
    OIDN_INLINE size_t getNumElements() const
    {
      if (dims.empty())
        return 0;
      size_t num = 1;
      for (size_t i = 0; i < dims.size(); ++i)
        num *= size_t(dims[i]);
      return num;
    }

    // Returns the size in bytes of the tensor
    OIDN_INLINE size_t getByteSize() const
    {
      if (paddedDims.empty())
        return 0;
      size_t num = 1;
      for (size_t i = 0; i < paddedDims.size(); ++i)
        num *= size_t(paddedDims[i]);
      return num * getDataTypeSize(dataType);
    }

    bool operator ==(const TensorDesc& other) const
    {
      return (dims == other.dims) && (paddedDims == other.paddedDims) &&
             (layout == other.layout) && (dataType == other.dataType);
    }

    bool operator !=(const TensorDesc& other) const
    {
      return (dims != other.dims) || (paddedDims != other.paddedDims) ||
             (layout != other.layout) || (dataType != other.dataType);
    }
  };

  // Tensor
  class Tensor : public Memory, protected TensorDesc
  {
  public:
    virtual void* getPtr() const = 0;

    OIDN_INLINE const TensorDesc& getDesc() const { return *this; }
    OIDN_INLINE const TensorDims& getDims() const { return dims; }
    OIDN_INLINE TensorLayout getLayout() const { return layout; }
    OIDN_INLINE DataType getDataType() const { return dataType; }

    using TensorDesc::getRank;
    using TensorDesc::getX;
    using TensorDesc::getPaddedX;
    using TensorDesc::getO;
    using TensorDesc::getPaddedO;
    using TensorDesc::getI;
    using TensorDesc::getPaddedI;
    using TensorDesc::getC;
    using TensorDesc::getPaddedC;
    using TensorDesc::getH;
    using TensorDesc::getW;
    using TensorDesc::getNumElements;
    using TensorDesc::getByteSize;

    template<typename T>
    operator TensorAccessor1D<T>()
    {
      if (layout != TensorLayout::x || dataType != DataTypeOf<T>::value)
        throw std::logic_error("incompatible tensor accessor");
      return TensorAccessor1D<T>(getPtr(), dims[0]);
    }

    template<typename T, TensorLayout accessorLayout>
    operator TensorAccessor3D<T, accessorLayout>()
    {
      if (layout != accessorLayout || dataType != DataTypeOf<T>::value)
        throw std::logic_error("incompatible tensor accessor");
      return TensorAccessor3D<T, accessorLayout>(getPtr(), getPaddedC(), getH(), getW());
    }

    template<typename T, TensorLayout accessorLayout>
    operator TensorAccessor4D<T, accessorLayout>()
    {
      if (layout != accessorLayout || dataType != DataTypeOf<T>::value)
        throw std::logic_error("incompatible tensor accessor");
      return TensorAccessor4D<T, accessorLayout>(getPtr(), getPaddedO(), getPaddedI(), getH(), getW());
    }

    std::shared_ptr<Tensor> map(Access access);

    void dump(const std::string& filenamePrefix);

  protected:
    explicit Tensor(const TensorDesc& desc);
    Tensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset = 0);

  private:
    template<typename T, TensorLayout layout>
    void dumpImpl(const std::string& filenamePrefix);
  };

  class GenericTensor final : public Tensor
  {
  public:
    explicit GenericTensor(const TensorDesc& desc);
    GenericTensor(const TensorDesc& desc, void* data);
    GenericTensor(const Ref<Engine>& engine, const TensorDesc& desc, Storage storage);
    GenericTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset = 0);
    ~GenericTensor();

    void* getPtr() const override { return ptr; }

  private:
    void updatePtr() override;

    void* ptr;   // pointer to the tensor data
    bool shared; // data owned and shared by someone else (buffer or user pointer)
  };

  using TensorMap = std::unordered_map<std::string, std::shared_ptr<Tensor>>;

OIDN_NAMESPACE_END
