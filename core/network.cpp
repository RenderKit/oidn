// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "network.h"
#include "conv.h"
#include "pool.h"
#include "upsample.h"
#include "color.h"

namespace oidn {

  Network::Network(const Ref<Device>& device, const std::shared_ptr<Weights>& weights)
    : device(device),
      weights(weights) {}

  std::shared_ptr<InputProcess> Network::addInputProcess(const std::string& name,
                                                         const TensorDims& srcDims,
                                                         int alignment,
                                                         const std::shared_ptr<TransferFunction>& transferFunc,
                                                         bool hdr,
                                                         bool snorm)
  {
    auto op = device->newInputProcess({srcDims, alignment, transferFunc, hdr, snorm});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  std::shared_ptr<OutputProcess> Network::addOutputProcess(const std::string& name,
                                                           const TensorDesc& srcDesc,
                                                           const std::shared_ptr<TransferFunction>& transferFunc,
                                                           bool hdr,
                                                           bool snorm)
  {
    auto op = device->newOutputProcess({srcDesc, transferFunc, hdr, snorm});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  std::shared_ptr<Conv> Network::addConv(const std::string& name,
                                         const TensorDesc& srcDesc,
                                         bool relu)
  {
    assert(weights);
    auto weight = weights->get(name + ".weight");
    auto bias   = weights->get(name + ".bias");

    auto op = device->newConv({srcDesc, weight, bias, relu});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

   std::shared_ptr<ConcatConv> Network::addConcatConv(const std::string& name,
                                                      const TensorDesc& src1Desc,
                                                      const TensorDesc& src2Desc,
                                                      bool relu)
  {
    assert(weights);
    auto weight = weights->get(name + ".weight");
    auto bias   = weights->get(name + ".bias");

    auto op = device->newConcatConv({src1Desc, src2Desc, weight, bias, relu});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  std::shared_ptr<Pool> Network::addPool(const std::string& name,
                                         const TensorDesc& srcDesc)
  {
    auto op = device->newPool({srcDesc});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  std::shared_ptr<Upsample> Network::addUpsample(const std::string& name,
                                                 const TensorDesc& srcDesc)
  {
    auto op = device->newUpsample({srcDesc});
    op->setName(name);
    ops.push_back(op);
    return op;
  }

  double Network::getWorkAmount() const
  {
    return double(ops.size());
  }

  bool Network::isSupported() const
  {
    for (const auto& op : ops)
      if (!op->isSupported())
        return false;
    return true;
  }

  size_t Network::getScratchByteSize() const
  {
    size_t scratchByteSize = 0;
    for (const auto& op : ops)
      scratchByteSize = max(scratchByteSize, op->getScratchByteSize());
    return scratchByteSize;
  }

  void Network::setScratch(const std::shared_ptr<Tensor>& scratch)
  {
    for (auto& op : ops)
      op->setScratch(scratch);
  }

  void Network::clear()
  {
    ops.clear();
  }

  void Network::finalize()
  {
    for (auto& op : ops)
      op->finalize();

    weights.reset();
  }

  void Network::run(Progress& progress)
  {
    for (size_t i = 0; i < ops.size(); ++i)
    {
      ops[i]->run();
      
      // Dump
      /*
      //device->wait();
      auto dst = ops[i]->getDst();
      if (dst)
        dst->dump("gpu/" + ops[i]->getName() + "_");
      */

      progress.update(1);
    }
  }

} // namespace oidn
