// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "bnns_node.h"

namespace oidn {

  BNNSNode::BNNSNode(const Ref<BNNSDevice>& device, const std::string& name)
    : BaseNode(device, name) {}

  BNNSNode::~BNNSNode()
  {
    if (filter)
      BNNSFilterDestroy(filter);
  }

  BNNSNDArrayDescriptor BNNSNode::toNDArrayDesc(Tensor& tz)
  {
    BNNSNDArrayDescriptor res;

    switch (tz.layout)
    {
    case TensorLayout::x:
      assert(ndims() == 1);
      res = BNNSNDArrayDescriptor({
        .layout = BNNSDataLayoutVector,
        .size   = {size_t(tz.dims[0])}
      });
      break;
    case TensorLayout::chw:
      assert(ndims() == 3);
      res = BNNSNDArrayDescriptor({
        .layout = BNNSDataLayoutImageCHW,
        .size   = {size_t(tz.dims[2]), size_t(tz.dims[1]), size_t(tz.dims[0])}
      });
      break;
    case TensorLayout::oihw:
      assert(ndims() == 4);
      res = BNNSNDArrayDescriptor({
        .layout = BNNSDataLayoutConvolutionWeightsOIHW,
        .size   = {size_t(tz.dims[3]), size_t(tz.dims[2]), size_t(tz.dims[1]), size_t(tz.dims[0])}
      });
      break;
    default:
      throw Exception(Error::Unknown, "invalid tensor layout");
    }

    switch (tz.dataType)
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
      throw Exception(Error::Unknown, "invalid tensor data type");
    }

    res.data = tz.data();
    return res;
  }

} // namespace oidn