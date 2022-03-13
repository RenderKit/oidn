// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "cuda_common.h"

namespace oidn {

  void checkError(cudnnStatus_t status)
  {
    if (status == CUDNN_STATUS_SUCCESS)
      return;

    const char* str = cudnnGetErrorString(status);
    switch (status)
    {
    case CUDNN_STATUS_ALLOC_FAILED:
      throw Exception(Error::OutOfMemory, str);
    case CUDNN_STATUS_ARCH_MISMATCH:
    case CUDNN_STATUS_NOT_SUPPORTED:
      throw Exception(Error::UnsupportedHardware, str);
    default:
      throw Exception(Error::Unknown, str);
    }
  }

  cudnnDataType_t toCuDNN(DataType dataType)
  {
    switch (dataType)
    {
    case DataType::Float32:
      return CUDNN_DATA_FLOAT;
    case DataType::Float16:
      return CUDNN_DATA_HALF;
    case DataType::UInt8:
      return CUDNN_DATA_UINT8;
    default:
      throw std::invalid_argument("unsupported data type");
    }
  }

  cudnnTensorDescriptor_t toCuDNNTensor(const TensorDesc& td)
  {
    // cuDNN supports tensors with only up to 2G elements
    if (td.getNumElements() > std::numeric_limits<int32_t>::max())
      return nullptr;

    cudnnTensorFormat_t cuFormat;
    int64_t H, W;

    switch (td.layout)
    {
    case TensorLayout::x:
      cuFormat = CUDNN_TENSOR_NCHW;
      H = 1;
      W = 1;
      break;
    case TensorLayout::chw:
      cuFormat = CUDNN_TENSOR_NCHW;
      H = td.dims[1];
      W = td.dims[2];
      break;
    case TensorLayout::hwc:
      cuFormat = CUDNN_TENSOR_NHWC;
      H = td.dims[1];
      W = td.dims[2];
      break;
    default:
      throw std::invalid_argument("unsupported tensor layout");
    }

    cudnnTensorDescriptor_t cuDesc;
    checkError(cudnnCreateTensorDescriptor(&cuDesc));
    checkError(cudnnSetTensor4dDescriptor(
      cuDesc,
      cuFormat,
      toCuDNN(td.dataType),
      1, int(td.dims[0]), int(H), int(W)));
    return cuDesc;
  }

  cudnnFilterDescriptor_t toCuDNNFilter(const TensorDesc& td)
  {
    cudnnTensorFormat_t cuFormat;
    if (td.layout == TensorLayout::oihw)
      cuFormat = CUDNN_TENSOR_NCHW;
    else if (td.layout == TensorLayout::ohwi)
      cuFormat = CUDNN_TENSOR_NHWC;
    else
      throw std::invalid_argument("unsupported filter layout");

    cudnnFilterDescriptor_t cuDesc;
    checkError(cudnnCreateFilterDescriptor(&cuDesc));
    checkError(cudnnSetFilter4dDescriptor(
      cuDesc,
      toCuDNN(td.dataType),
      cuFormat,
      int(td.dims[0]), int(td.dims[1]), int(td.dims[2]), int(td.dims[3])));
    return cuDesc;
  }

} // namespace oidn