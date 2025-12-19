// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cpu_common.h"
#if !defined(OIDN_DNNL) && !defined(OIDN_BNNS)
  #include "cpu_conv_ispc.h"
  #if defined(OIDN_ARCH_X64)
    #include "cpu_conv_amx_ispc.h"
  #endif
#endif

OIDN_NAMESPACE_BEGIN

  Image::operator ispc::ImageAccessor()
  {
    ispc::ImageAccessor acc;
    acc.ptr = reinterpret_cast<int8_t*>(ptr);
    acc.hByteStride = hByteStride;
    acc.wByteStride = wByteStride;

    switch (getDataType())
    {
    case DataType::Undefined:
      acc.dataType = ispc::DataType_Undefined;
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

#if !defined(OIDN_BNNS)
  Tensor::operator ispc::TensorAccessor3D_ChwBc()
  {
    if (getRank() != 3 || (layout != TensorLayout::Chw8c  &&
                           layout != TensorLayout::Chw16c &&
                           layout != TensorLayout::Chw32c))
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor3D_ChwBc acc;
    acc.ptr = static_cast<int8_t*>(getPtr());
    acc.C = getPaddedC();
    acc.H = getH();
    acc.W = getW();

    const int B = getTensorLayoutInfo(layout).blockC;
    const uint32_t cByteStride = uint32_t(getDataTypeSize(dataType));
    const uint32_t wByteStride = uint32_t(B) * cByteStride;
    acc.hByteStride = uint32_t(getW()) * wByteStride;
    acc.CByteStride = uint32_t(getH()) * acc.hByteStride;

    return acc;
  }
#else
  Tensor::operator ispc::TensorAccessor3D_chw()
  {
    if (getRank() != 3 || layout != TensorLayout::chw)
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor3D_chw acc;
    acc.ptr = static_cast<int8_t*>(getPtr());
    acc.C = getPaddedC();
    acc.H = getH();
    acc.W = getW();

    const uint32_t wByteStride = uint32_t(getDataTypeSize(dataType));
    acc.hByteStride = uint32_t(getW()) * wByteStride;
    acc.cByteStride = uint32_t(getH()) * acc.hByteStride;

    return acc;
  }
#endif

#if !defined(OIDN_DNNL) && !defined(OIDN_BNNS)
  Tensor::operator ispc::TensorAccessor1D()
  {
    if (layout != TensorLayout::x)
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor1D acc;
    acc.ptr = static_cast<int8_t*>(getPtr());
    acc.X = getPaddedX();
    return acc;
  }

  Tensor::operator ispc::TensorAccessor4D_IOhwBiBo()
  {
    if (getRank() != 4 || (layout != TensorLayout::IOhw8i8o && layout != TensorLayout::IOhw16i16o))
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor4D_IOhwBiBo acc;
    acc.ptr = static_cast<int8_t*>(getPtr());

    acc.O = getPaddedO();
    acc.I = getPaddedI();
    acc.H = getH();
    acc.W = getW();

    const int B = getTensorLayoutInfo(layout).blockC;
    const uint32_t BoByteStride = uint32_t(getDataTypeSize(dataType));
    const uint32_t BiByteStride = uint32_t(B) * BoByteStride;
    const uint32_t wByteStride  = uint32_t(B) * BiByteStride;
    acc.hByteStride  = uint32_t(getW()) * wByteStride;
    acc.OByteStride  = uint32_t(getH()) * acc.hByteStride;
    acc.IByteStride  = uint32_t(getPaddedO() / B) * acc.OByteStride;

    return acc;
  }

#if defined(OIDN_ARCH_X64)
  Tensor::operator ispc::TensorAccessor4D_OIhwPoQiRoSi()
  {
    if (getRank() != 4 || layout != TensorLayout::OIhw2o16i16o2i)
      throw std::logic_error("incompatible tensor accessor");

    ispc::TensorAccessor4D_OIhwPoQiRoSi acc;
    acc.ptr = static_cast<int8_t*>(getPtr());

    const int B = getTensorLayoutInfo(layout).blockC;
    const int P = 2;
    const int Q = 16;
    const int R = 16;
    const int S = 2;

    const uint32_t SiByteStride = uint32_t(getDataTypeSize(dataType));
    const uint32_t RoByteStride = uint32_t(S * SiByteStride);
    const uint32_t QiByteStride = uint32_t(R * RoByteStride);
    const uint32_t PoByteStride = uint32_t(Q * QiByteStride);
    const uint32_t wByteStride  = uint32_t(P * PoByteStride);
    const uint32_t hByteStride = uint32_t(getW()) * wByteStride;
    const uint32_t IByteStride = uint32_t(getH()) * hByteStride;
    acc.OByteStride = uint32_t(getPaddedI() / B) * IByteStride;

    return acc;
  }
#endif
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