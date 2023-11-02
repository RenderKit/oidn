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

    switch (desc.getDataType())
    {
    case DataType::Void:
      acc.dataType = ispc::DataType_Void;
      break;
    /*
    case DataType::UInt8:
      acc.dataType = ispc::DataType_UInt8;
      break;
    */
    case DataType::Float16:
      acc.dataType = ispc::DataType_Float16;
      break;
    case DataType::Float32:
      acc.dataType = ispc::DataType_Float32;
      break;
    default:
      throw std::logic_error("unsupported data type");
    }

    acc.C = desc.getC();
    if (acc.C > 3)
      throw std::logic_error("unsupported number of channels for image accessor");
    acc.H = desc.getH();
    acc.W = desc.getW();

    return acc;
  }

  template<> ispc::TensorAccessor3D toISPC<ispc::TensorAccessor3D>(Tensor& tensor)
  {
    if (tensor.getRank() != 3 || tensor.getDataType() != DataType::Float32)
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor3D acc3;
    acc3.ptr = static_cast<float*>(tensor.getData());
    acc3.C = tensor.getPaddedC();
    acc3.H = tensor.getH();
    acc3.W = tensor.getW();
    return acc3;
  }

#if defined(OIDN_ISPC)
  template<> ispc::TensorAccessor1D toISPC<ispc::TensorAccessor1D>(Tensor& tensor)
  {
    if (tensor.getRank() != 1 || tensor.getDataType() != DataType::Float32)
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor1D acc1;
    acc1.ptr = static_cast<float*>(tensor.getData());
    acc1.X = tensor.getPaddedX();
    return acc1;
  }

  template<> ispc::TensorAccessor4D toISPC<ispc::TensorAccessor4D>(Tensor& tensor)
  {
    if (tensor.getRank() != 4 || tensor.getDataType() != DataType::Float32)
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor4D acc4;
    acc4.ptr = static_cast<float*>(tensor.getData());
    acc4.O = tensor.getPaddedO();
    acc4.I = tensor.getPaddedI();
    acc4.H = tensor.getH();
    acc4.W = tensor.getW();
    return acc4;
  }
#endif

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