// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "buffer.h"
#include "tensor_desc.h"
#include "tensor_accessor.h"
#include <unordered_map>
#include <iostream>

OIDN_NAMESPACE_BEGIN

  class Engine;

  class Tensor : public Memory, protected TensorDesc
  {
  public:
    virtual void* getPtr() const = 0;

    oidn_inline const TensorDesc& getDesc() const { return *this; }
    oidn_inline const TensorDims& getDims() const { return dims; }
    oidn_inline TensorLayout getLayout() const { return layout; }
    oidn_inline DataType getDataType() const { return dataType; }

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
      return TensorAccessor1D<T>(getPtr(), getPaddedX());
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

    operator ispc::TensorAccessor1D();
    operator ispc::TensorAccessor3D_chw();
    operator ispc::TensorAccessor3D_ChwBc();
    operator ispc::TensorAccessor4D_IOhwBiBo();
    operator ispc::TensorAccessor4D_OIhwPoQiRoSi();

    virtual Ref<Tensor> toDevice(Engine* engine, Storage storage = Storage::Device);

    // Debug
  #if 0
    uint32_t getHash() const;
    void dump(const std::string& filenamePrefix);
  #endif

  protected:
    explicit Tensor(const TensorDesc& desc);
    Tensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset = 0);

  private:
    template<typename T, TensorLayout layout>
    void dumpImpl(const std::string& filenamePrefix);
  };

  class HostTensor final : public Tensor
  {
  public:
    explicit HostTensor(const TensorDesc& desc);
    HostTensor(const TensorDesc& desc, void* data);
    ~HostTensor();

    void* getPtr() const override { return ptr; }

    Ref<Tensor> toDevice(Engine* engine, Storage storage = Storage::Device) override;

  private:
    void* ptr;   // pointer to the tensor data
    bool shared; // data owned and shared by the user
  };

  class DeviceTensor final : public Tensor
  {
  public:
    DeviceTensor(Engine* engine, const TensorDesc& desc, Storage storage);
    DeviceTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset = 0);

    void* getPtr() const override { return ptr; }

  private:
    void postRealloc() override;

    void* ptr; // pointer to the tensor data
  };

  using TensorMap = std::unordered_map<std::string, Ref<Tensor>>;

OIDN_NAMESPACE_END
