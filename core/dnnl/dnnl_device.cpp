// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_device.h"
#include "dnnl_tensor.h"
#include "dnnl_conv.h"
#include "dnnl_pool.h"

namespace oidn {

  void DNNLDevice::wait()
  {
    dnnlStream.wait();
  }

  std::shared_ptr<Tensor> DNNLDevice::newTensor(const TensorDesc& desc)
  {
    return std::make_shared<DNNLTensor>(this, desc);
  }

  std::shared_ptr<Tensor> DNNLDevice::newTensor(const TensorDesc& desc, void* data)
  {
    return std::make_shared<DNNLTensor>(this, desc, data);
  }

  std::shared_ptr<Tensor> DNNLDevice::newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
  {
    assert(buffer->getDevice() == this);
    return std::make_shared<DNNLTensor>(buffer, desc, byteOffset);
  }

  std::shared_ptr<Conv> DNNLDevice::newConv(const ConvDesc& desc)
  {
    return std::make_shared<DNNLConv>(this, desc);
  }

  std::shared_ptr<Pool> DNNLDevice::newPool(const PoolDesc& desc)
  {
    return std::make_shared<DNNLPool>(this, desc);
  }

} // namespace oidn
