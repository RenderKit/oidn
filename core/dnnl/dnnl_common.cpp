// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_common.h"
#include "dnnl_tensor.h"

namespace oidn {

  dnnl::memory::desc toDNNL(const TensorDesc& td)
  {
    dnnl::memory::dims dnnlDims;
    dnnl::memory::format_tag dnnlFormat;
    switch (td.layout)
    {
    case TensorLayout::x:
      assert(td.getRank() == 1);
      dnnlDims   = {td.dims[0]};
      dnnlFormat = dnnl::memory::format_tag::x;
      break;
    case TensorLayout::chw:
      assert(td.getRank() == 3);
      dnnlDims   = {1, td.dims[0], td.dims[1], td.dims[2]};
      dnnlFormat = dnnl::memory::format_tag::nchw;
      break;
    case TensorLayout::Chw8c:
      assert(td.getRank() == 3);
      dnnlDims   = {1, td.dims[0], td.dims[1], td.dims[2]};
      dnnlFormat = dnnl::memory::format_tag::nChw8c;
      break;
    case TensorLayout::Chw16c:
      assert(td.getRank() == 3);
      dnnlDims   = {1, td.dims[0], td.dims[1], td.dims[2]};
      dnnlFormat = dnnl::memory::format_tag::nChw16c;
      break;
    case TensorLayout::oihw:
      assert(td.getRank() == 4);
      dnnlDims   = {td.dims[0], td.dims[1], td.dims[2], td.dims[3]};
      dnnlFormat = dnnl::memory::format_tag::oihw;
      break;
    default:
      throw Exception(Error::Unknown, "unsupported tensor layout");
    }

    dnnl::memory::data_type dnnlType;
    switch (td.dataType)
    {
    case DataType::Float32:
      dnnlType = dnnl::memory::data_type::f32;
      break;
    case DataType::Float16:
      dnnlType = dnnl::memory::data_type::f16;
      break;
    case DataType::UInt8:
      dnnlType = dnnl::memory::data_type::u8;
      break;
    default:
      throw Exception(Error::Unknown, "unsupported data type");
    }

    return dnnl::memory::desc(dnnlDims, dnnlType, dnnlFormat);
  }

  const dnnl::memory& getDNNL(const Tensor& tensor)
  {
    if (auto dnnlTensor = dynamic_cast<const DNNLTensor*>(&tensor))
      return dnnlTensor->getDNNLMemory();
    else
      throw Exception(Error::Unknown, "not DNNLTensor");
  }

} // namespace oidn