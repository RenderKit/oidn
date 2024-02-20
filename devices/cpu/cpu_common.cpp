// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_common.h"
#if !defined(OIDN_DNNL) && !defined(OIDN_BNNS)
  #include "cpu_conv_ispc.h"
#endif

OIDN_NAMESPACE_BEGIN

  Image::operator ispc::ImageAccessor()
  {
    ispc::ImageAccessor acc;
    acc.ptr = reinterpret_cast<uint8_t*>(ptr);
    acc.hByteStride = hByteStride;
    acc.wByteStride = wByteStride;

    switch (getDataType())
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

    acc.C = getC();
    if (acc.C > 3)
      throw std::logic_error("unsupported number of channels for image accessor");
    acc.H = getH();
    acc.W = getW();

    return acc;
  }

  Tensor::operator ispc::TensorAccessor3D()
  {
    if (getRank() != 3 || layout == TensorLayout::hwc)
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor3D acc;
    acc.ptr = static_cast<uint8_t*>(getPtr());
    acc.C = getPaddedC();
    acc.H = getH();
    acc.W = getW();

    const int B = getTensorLayoutInfo(layout).blockC;
    const size_t cByteStride = getDataTypeSize(dataType);
    const size_t wByteStride = B * cByteStride;
    acc.hByteStride = size_t(acc.W) * wByteStride;
    acc.CByteStride = size_t(acc.H) * acc.hByteStride;

    return acc;
  }

#if !defined(OIDN_DNNL) && !defined(OIDN_BNNS)
  Tensor::operator ispc::TensorAccessor1D()
  {
    if (layout != TensorLayout::x)
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor1D acc;
    acc.ptr = static_cast<uint8_t*>(getPtr());
    acc.X = getPaddedX();
    return acc;
  }

  Tensor::operator ispc::TensorAccessor4D()
  {
    if (getRank() != 4 || (layout != TensorLayout::IOhw8i8o && layout != TensorLayout::IOhw16i16o))
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor4D acc;
    acc.ptr = static_cast<uint8_t*>(getPtr());

    acc.O = getPaddedO();
    acc.I = getPaddedI();
    acc.H = getH();
    acc.W = getW();

    const int B = getTensorLayoutInfo(layout).blockC;
    const size_t BoByteStride = getDataTypeSize(dataType);
    const size_t BiByteStride = B * BoByteStride;
    const size_t wByteStride  = B * BiByteStride;
    acc.hByteStride = size_t(acc.W)     * wByteStride;
    acc.OByteStride = size_t(acc.H)     * acc.hByteStride;
    acc.IByteStride = size_t(acc.O / B) * acc.OByteStride;

    return acc;
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

    res.inputScalePtr = tf.inputScalePtr;
    res.inputScale    = tf.inputScale;
    res.outputScale   = tf.outputScale;

    return res;
  }

OIDN_NAMESPACE_END