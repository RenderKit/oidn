// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_common.h"
#include "dnnl_tensor.h"

namespace oidn {

  dnnl::memory::data_type toDNNL(DataType dt)
  {
    switch (dt)
    {
    case DataType::Float32:
      return dnnl::memory::data_type::f32;
    case DataType::Float16:
      return dnnl::memory::data_type::f16;
    case DataType::UInt8:
      return dnnl::memory::data_type::u8;
    default:
      throw Exception(Error::Unknown, "unsupported data type");
    }
  }

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

    return dnnl::memory::desc(dnnlDims, toDNNL(td.dataType), dnnlFormat);
  }

  const dnnl::memory& getDNNL(const std::shared_ptr<Tensor>& tensor)
  {
    if (auto dnnlTensor = dynamic_cast<const DNNLTensor*>(tensor.get()))
      return dnnlTensor->getDNNLMemory();
    else
      throw Exception(Error::Unknown, "not DNNLTensor");
  }

} // namespace oidn