// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "mps/mps_network.h"
#include "mps/mps_node.h"

namespace oidn {

  MPSNetwork::MPSNetwork(const Ref<Device>& device, const std::map<std::string, std::shared_ptr<Tensor>>& weightsMap)
    : Network(device, weightsMap)
  {
    mpsNet.reset(new MPSGraphNetworkImpl());
    mpsNet->init();
    mpsNet->createGraph();
  }

  void MPSNetwork::execute(Progress& progress)
  {
    inputReorder->execute();
    progress.update(1);
    mpsNet->execute(inputReorder->getDst()->data(), inputReorder->getDst()->data());
    progress.update(1);
    outputReorder->execute();
    progress.update(1);
  }

  double MPSNetwork::getWorkAmount() const
  {
    return double(3);
  }

  std::shared_ptr<Tensor> MPSNetwork::newTensor(const TensorDesc& desc, ptrdiff_t offset)
  {
    return Network::newTensor(desc, 0);
  }

  std::shared_ptr<InputReorderNode> MPSNetwork::addInputReorder(const std::string& name,
                                                             const std::shared_ptr<Tensor>& dst,
                                                             const std::shared_ptr<TransferFunction>& transferFunc,
                                                             bool hdr,
                                                             bool snorm)
  {
    
    std::vector<int64_t> dims = dst->desc().dims;
    dims.insert(dims.begin(), 1);
    
    mpsNet->addInput(dims, name);
    
    tensorToName[dst] = name;
    inputReorder = Network::addInputReorder(name, dst, transferFunc, hdr, snorm);
    return inputReorder;
  }


  std::shared_ptr<OutputReorderNode> MPSNetwork::addOutputReorder(const std::string& name,
                                                               const std::shared_ptr<Tensor>& src,
                                                               const std::shared_ptr<TransferFunction>& transferFunc,
                                                               bool hdr,
                                                               bool snorm)
  {
    std::string outName = tensorToName[src];
    
    mpsNet->setOutputs({outName});
    
    outputReorder = Network::addOutputReorder(name, inputReorder->getDst(), transferFunc, hdr, snorm);
    return outputReorder;
  }

  std::shared_ptr<Node> MPSNetwork::addConv(const std::string& name,
                                         const std::shared_ptr<Tensor>& src,
                                         const std::shared_ptr<Tensor>& dst,
                                         bool relu)
  {
    assert(dst->desc() == getConvDesc(name, src->desc()));

    // Get and pad the weights
    auto weights = weightsMap[name + ".weight"];
    if (weights->ndims() != 4 || weights->layout != TensorLayout::oihw)
      throw Exception(Error::InvalidOperation, "invalid convolution weights");

    // Get and pad the biases
    auto bias = weightsMap[name + ".bias"];
    if (bias->ndims() != 1)
      throw Exception(Error::InvalidOperation, "invalid convolution biases");

    std::vector<int64_t> strides = {1, 1};
    std::vector<int64_t> padding = {1, 1};
    std::vector<int64_t> dilation = {1, 1};
    
    std::string convName = name + "_conv";
    std::string biasName = name + "_bias";
    
    std::string srcName = tensorToName[src];
    
    mpsNet->addConv(srcName, 1, padding, strides, dilation, weights->dims, weights->data(), convName);
    
    std::vector<int64_t> biasShape = {1, bias->dims[0], 1, 1};
    
    if (relu) {
      mpsNet->addAdd(convName, biasShape, bias->data(), biasName);
      mpsNet->addRelu(biasName, name);
    } else {
      mpsNet->addAdd(convName, biasShape, bias->data(), name);
    }
    
    tensorToName[dst] = name;
        
    // Create the convolution node
    auto node = std::make_shared<MPSNode>(device, name, dst);
    return node;
  }


  std::shared_ptr<Node> MPSNetwork::addPool(const std::string& name,
                                         const std::shared_ptr<Tensor>& src,
                                         const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->desc() == getPoolDesc(src->desc()));

    std::vector<int64_t> strides = {2, 2};
    std::vector<int64_t> padding = {0, 0};
    std::vector<int64_t> kernel = {2, 2};
    
    std::string srcName = tensorToName[src];
    
    mpsNet->addPool(srcName, kernel, padding, strides, name);
    
    tensorToName[dst] = name;
    
    auto node = std::make_shared<MPSNode>(device, name, dst);
    return node;
  }

  std::shared_ptr<Node> MPSNetwork::addUpsample(const std::string& name,
                                             const std::shared_ptr<Tensor>& src,
                                             const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->desc() == getUpsampleDesc(src->desc()));
    
    std::string srcName = tensorToName[src];
    
    std::vector<int64_t> dims = { dst->desc().dims[1], dst->desc().dims[2] };
    
    mpsNet->addUpsample(srcName, dims, name);

    tensorToName[dst] = name;
    
    auto node = std::make_shared<MPSNode>(device, name, dst);
    return node;
  }

  std::shared_ptr<Node> MPSNetwork::addConcat(const std::string& name,
                                const std::shared_ptr<Tensor>& src1,
                                const std::shared_ptr<Tensor>& src2,
                                const std::shared_ptr<Tensor>& dst) {
    std::vector<std::string> srcNames = {tensorToName[src1], tensorToName[src2]};
    
    mpsNet->addConcat(srcNames, name);
    
    tensorToName[dst] = name;
    
    auto node = std::make_shared<MPSNode>(device, name, dst);
    return node;
  }

} // namespace oidn
