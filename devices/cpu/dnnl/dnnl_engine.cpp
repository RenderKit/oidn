// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "dnnl_engine.h"
#include "dnnl_tensor.h"
#include "dnnl_conv.h"

OIDN_NAMESPACE_BEGIN

  DNNLEngine::DNNLEngine(CPUDevice* device, int numThreads)
    : CPUEngine(device, numThreads)
  {
    dnnl_set_verbose(clamp(device->verbose - 2, 0, 2)); // unfortunately this is not per-device but global
    dnnlEngine = dnnl::engine(dnnl::engine::kind::cpu, 0);
    dnnlStream = dnnl::stream(dnnlEngine);
  }

  Ref<Tensor> DNNLEngine::newTensor(const TensorDesc& desc, Storage storage)
  {
    if (!isSupported(desc))
      throw std::invalid_argument("unsupported tensor descriptor");

    return makeRef<DNNLTensor>(this, desc, storage);
  }

  Ref<Tensor> DNNLEngine::newTensor(const Ref<Buffer>& buffer, const TensorDesc& desc, size_t byteOffset)
  {
    if (!isSupported(desc))
      throw std::invalid_argument("unsupported tensor descriptor");
    if (buffer->getEngine() != this)
      throw std::invalid_argument("buffer was created by a different engine");

    return makeRef<DNNLTensor>(buffer, desc, byteOffset);
  }

  Ref<Conv> DNNLEngine::newConv(const ConvDesc& desc)
  {
    return makeRef<DNNLConv>(this, desc);
  }

OIDN_NAMESPACE_END
