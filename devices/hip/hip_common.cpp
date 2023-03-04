// Copyright 2009-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "hip_common.h"

OIDN_NAMESPACE_BEGIN

  void checkError(miopenStatus_t status)
  {
    if (status == miopenStatusSuccess)
      return;

    const char* str = miopenGetErrorString(status);
    switch (status)
    {
    case miopenStatusAllocFailed:
      throw Exception(Error::OutOfMemory, str);
    /*
    case MIOPEN_STATUS_ARCH_MISMATCH:
    case MIOPEN_STATUS_NOT_SUPPORTED:
      throw Exception(Error::UnsupportedHardware, str);
    */
    default:
      throw Exception(Error::Unknown, str);
    }
  }

  miopenDataType_t toMIOpen(DataType dataType)
  {
    switch (dataType)
    {
    case DataType::Float32:
      return miopenFloat;
    case DataType::Float16:
      return miopenHalf;
    case DataType::UInt8:
      return miopenInt8; // FIXME
    default:
      throw std::invalid_argument("unsupported data type");
    }
  }

  miopenTensorDescriptor_t toMIOpen(const TensorDesc& td)
  {
    miopenTensorDescriptor_t miDesc;
    checkError(miopenCreateTensorDescriptor(&miDesc));

    switch (td.layout)
    {
    case TensorLayout::x:
      checkError(miopenSet4dTensorDescriptor(
        miDesc,
        toMIOpen(td.dataType),
        1, td.getX(), 1, 1));
      break;
      
    case TensorLayout::chw:
      checkError(miopenSet4dTensorDescriptor(
        miDesc,
        toMIOpen(td.dataType),
        1, td.getPaddedC(), td.getH(), td.getW()));
      break;

    case TensorLayout::oihw:
      checkError(miopenSet4dTensorDescriptor(
        miDesc,
        toMIOpen(td.dataType),
        td.getPaddedO(), td.getPaddedI(), td.getH(), td.getW()));
      break;

    default:
      throw std::invalid_argument("unsupported tensor layout");
    }

    return miDesc;
  }

OIDN_NAMESPACE_END