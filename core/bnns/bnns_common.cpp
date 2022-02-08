// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_common.h"

namespace oidn {

  BNNSNDArrayDescriptor toBNNS(const Tensor& t)
  {
    BNNSNDArrayDescriptor res;

    switch (t.layout)
    {
    case TensorLayout::x:
      assert(getRank() == 1);
      res = BNNSNDArrayDescriptor({
        .layout = BNNSDataLayoutVector,
        .size   = {size_t(t.dims[0])}
      });
      break;
    case TensorLayout::chw:
      assert(getRank() == 3);
      res = BNNSNDArrayDescriptor({
        .layout = BNNSDataLayoutImageCHW,
        .size   = {size_t(t.dims[2]), size_t(t.dims[1]), size_t(t.dims[0])}
      });
      break;
    case TensorLayout::oihw:
      assert(getRank() == 4);
      res = BNNSNDArrayDescriptor({
        .layout = BNNSDataLayoutConvolutionWeightsOIHW,
        .size   = {size_t(t.dims[3]), size_t(t.dims[2]), size_t(t.dims[1]), size_t(t.dims[0])}
      });
      break;
    default:
      throw Exception(Error::Unknown, "unsupported tensor layout");
    }

    switch (t.dataType)
    {
    case DataType::Float32:
      res.data_type = BNNSDataTypeFloat32;
      break;
    case DataType::Float16:
      res.data_type = BNNSDataTypeFloat16;
      break;
    case DataType::UInt8:
      res.data_type = BNNSDataTypeUInt8;
      break;
    default:
      throw Exception(Error::Unknown, "unsupported data type");
    }

    res.data = (void*)t.getData();
    return res;
  }

} // namespace oidn