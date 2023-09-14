// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_common.h"

OIDN_NAMESPACE_BEGIN

  ispc::ImageAccessor toISPC(Image& image)
  {
    const ImageDesc& desc = image.getDesc();

    ispc::ImageAccessor acc;
    acc.ptr = static_cast<uint8_t*>(image.getData());
    acc.hByteStride = desc.hByteStride;
    acc.wByteStride = desc.wByteStride;

    if (desc.format != Format::Undefined)
    {
      switch (desc.getDataType())
      {
      case DataType::Float32:
        acc.dataType = ispc::DataType_Float32;
        break;
      case DataType::Float16:
        acc.dataType = ispc::DataType_Float16;
        break;
      case DataType::UInt8:
        acc.dataType = ispc::DataType_UInt8;
        break;
      default:
        throw std::logic_error("unsupported data type");
      }
    }
    else
      acc.dataType = ispc::DataType_Float32;

    acc.W = int(desc.width);
    acc.H = int(desc.height);

    return acc;
  }

  template<class U>
  U toISPC(Tensor& tensor)
  {
    if (tensor.getDataType() != DataType::Float32)
      throw std::logic_error("incompatible tensor accessor");

    switch(tensor.getLayout())
    {
      case TensorLayout::x:
        assert(td.getRank() == 1);
        if constexpr(std::is_same<U, ispc::TensorAccessor1D>::value)
        {
          ispc::TensorAccessor1D acc1;
          acc1.ptr = static_cast<float*>(tensor.getData());
          acc1.X = tensor.getPaddedX();
          return acc1;
        }
        else
        {
          throw std::logic_error("incompatible template and layout");
        }
      case TensorLayout::Chw8c:
        assert(td.getRank() == 3);
        if constexpr(std::is_same<U, ispc::TensorAccessor3D>::value)
        {
          ispc::TensorAccessor3D acc3;
          acc3.ptr = static_cast<float*>(tensor.getData());
          acc3.C = tensor.getPaddedC();
          acc3.H = tensor.getH();
          acc3.W = tensor.getW();
          return acc3;
        }
        else
        {
          throw std::logic_error("incompatible template and layout");
        }
      case TensorLayout::oihw:
        assert(td.getRank() == 4);
        if constexpr(std::is_same<U, ispc::TensorAccessor4D>::value)
        {
          ispc::TensorAccessor4D acc4;
          acc4.ptr = static_cast<float*>(tensor.getData());
          acc4.O = tensor.getPaddedO();
          acc4.I = tensor.getPaddedI();
          acc4.H = tensor.getH();
          acc4.W = tensor.getW();
          return acc4;
        }
        else
        {
          throw std::logic_error("incompatible template and layout");
        }
      default:
        throw std::invalid_argument("unsupported tensor layout");
    }
  }

  ispc::Tile toISPC(const Tile& tile)
  {
    ispc::Tile res;
    res.hSrcBegin = tile.hSrcBegin;
    res.wSrcBegin = tile.wSrcBegin;
    res.hDstBegin = tile.hDstBegin;
    res.wDstBegin = tile.wDstBegin;
    res.H = tile.H;
    res.W = tile.W;
    return res;
  }

  ispc::TransferFunction toISPC(const TransferFunction& tf)
  {
    ispc::TransferFunction res;

    switch (tf.getType())
    {
    case TransferFunction::Type::Linear: ispc::LinearTransferFunction_Constructor(&res); break;
    case TransferFunction::Type::SRGB:   ispc::SRGBTransferFunction_Constructor(&res);   break;
    case TransferFunction::Type::PU:     ispc::PUTransferFunction_Constructor(&res);     break;
    case TransferFunction::Type::Log:    ispc::LogTransferFunction_Constructor(&res);    break;
    default:
      assert(0);
    }

    res.inputScale  = tf.getInputScale();
    res.outputScale = tf.getOutputScale();

    return res;
  }

OIDN_NAMESPACE_END