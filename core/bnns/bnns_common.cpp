// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_common.h"

namespace oidn {

  BNNSNDArrayDescriptor toBNNS(const TensorDesc& td)
  {
    BNNSNDArrayDescriptor res;

    switch (td.layout)
    {
    case TensorLayout::x:
      assert(getRank() == 1);
      res = BNNSNDArrayDescriptor({
        .layout = BNNSDataLayoutVector,
        .size   = {size_t(td.dims[0])}
      });
      break;
    case TensorLayout::chw:
      assert(getRank() == 3);
      res = BNNSNDArrayDescriptor({
        .layout = BNNSDataLayoutImageCHW,
        .size   = {size_t(td.dims[2]), size_t(td.dims[1]), size_t(td.dims[0])}
      });
      break;
    case TensorLayout::oihw:
      assert(getRank() == 4);
      res = BNNSNDArrayDescriptor({
        .layout = BNNSDataLayoutConvolutionWeightsOIHW,
        .size   = {size_t(td.dims[3]), size_t(td.dims[2]), size_t(td.dims[1]), size_t(td.dims[0])}
      });
      break;
    default:
      throw std::invalid_argument("unsupported tensor layout");
    }

    switch (td.dataType)
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
      throw std::invalid_argument("unsupported data type");
    }

    res.data = nullptr;
    return res;
  }

  BNNSNDArrayDescriptor toBNNS(const std::shared_ptr<Tensor>& t)
  {
    BNNSNDArrayDescriptor res = toBNNS(t->getDesc());
    res.data = (void*)t->getData();
    return res;
  }

} // namespace oidnt,