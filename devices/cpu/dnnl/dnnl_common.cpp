// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_common.h"
#include "dnnl_tensor.h"

OIDN_NAMESPACE_BEGIN

  dnnl::memory::data_type toDNNL(DataType dataType)
  {
    switch (dataType)
    {
    case DataType::Float32:
      return dnnl::memory::data_type::f32;
    case DataType::Float16:
      return dnnl::memory::data_type::f16;
    case DataType::UInt8:
      return dnnl::memory::data_type::u8;
    default:
      throw std::invalid_argument("unsupported data type");
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
      dnnlDims   = {td.getPaddedX()};
      dnnlFormat = dnnl::memory::format_tag::x;
      break;
    case TensorLayout::chw:
      assert(td.getRank() == 3);
      dnnlDims   = {1, td.getPaddedC(), td.getH(), td.getW()};
      dnnlFormat = dnnl::memory::format_tag::nchw;
      break;
    case TensorLayout::Chw8c:
      assert(td.getRank() == 3);
      dnnlDims   = {1, td.getPaddedC(), td.getH(), td.getW()};
      dnnlFormat = dnnl::memory::format_tag::nChw8c;
      break;
    case TensorLayout::Chw16c:
      assert(td.getRank() == 3);
      dnnlDims   = {1, td.getPaddedC(), td.getH(), td.getW()};
      dnnlFormat = dnnl::memory::format_tag::nChw16c;
      break;
    case TensorLayout::oihw:
      assert(td.getRank() == 4);
      dnnlDims   = {td.getPaddedO(), td.getPaddedI(), td.getH(), td.getW()};
      dnnlFormat = dnnl::memory::format_tag::oihw;
      break;
    case TensorLayout::OIhw8i8o:
      assert(td.getRank() == 4);
      dnnlDims   = {td.getPaddedO(), td.getPaddedI(), td.getH(), td.getW()};
      dnnlFormat = dnnl::memory::format_tag::OIhw8i8o;
      break;
    case TensorLayout::OIhw16i16o:
      assert(td.getRank() == 4);
      dnnlDims   = {td.getPaddedO(), td.getPaddedI(), td.getH(), td.getW()};
      dnnlFormat = dnnl::memory::format_tag::OIhw16i16o;
      break;
    case TensorLayout::OIhw2o8i8o2i:
      assert(td.getRank() == 4);
      dnnlDims   = {td.getPaddedO(), td.getPaddedI(), td.getH(), td.getW()};
      dnnlFormat = dnnl::memory::format_tag::OIhw2o8i8o2i;
      break;
    default:
      throw std::invalid_argument("unsupported tensor layout");
    }

    return dnnl::memory::desc(dnnlDims, toDNNL(td.dataType), dnnlFormat);
  }

  dnnl::memory toDNNL(const Ref<Buffer>& buffer)
  {
    DNNLEngine* engine = dynamic_cast<DNNLEngine*>(buffer->getEngine());
    if (!engine)
      throw std::logic_error("buffer does not belong to a DNNL engine");
    dnnl::memory::desc desc({int64_t(buffer->getByteSize())}, dnnl::memory::data_type::u8, dnnl::memory::format_tag::x);
    return {desc, engine->getDNNLEngine(), buffer->getPtr()};
  }

  const dnnl::memory& getDNNL(const Ref<Tensor>& tensor)
  {
    if (auto dnnlTensor = dynamic_cast<const DNNLTensor*>(tensor.get()))
      return dnnlTensor->getDNNLMemory();
    else
      throw std::invalid_argument("not DNNLTensor");
  }

OIDN_NAMESPACE_END