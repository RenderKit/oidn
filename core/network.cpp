// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "conv.h"
#include "pool.h"
#include "upsample.h"
#include "color.h"
#include "network.h"

namespace oidn {

  Network::Network(const Ref<Device>& device, const std::map<std::string, std::shared_ptr<Tensor>>& weightsMap)
    : device(device),
      blockSize(device->getTensorBlockSize()),
      weightsMap(weightsMap)
  {
  }

  void Network::execute(Progress& progress)
  {
    for (size_t i = 0; i < nodes.size(); ++i)
    {
      nodes[i]->execute();
      
      // Dump
      /*
      //device->wait();
      auto dst = nodes[i]->getDst();
      if (dst)
        dst->dump("gpu/" + nodes[i]->getName() + "_");
      */

      progress.update(1);
    }
  }

  double Network::getWorkAmount() const
  {
    return double(nodes.size());
  }

  void Network::allocScratch(size_t size)
  {
    assert(!scratch);
    scratch = device->newScratchBuffer(size);
  }

  std::shared_ptr<Tensor> Network::newTensor(const TensorDesc& desc, ptrdiff_t offset)
  {
    assert(scratch);
    return scratch->newTensor(desc, offset);
  }

  std::shared_ptr<Image> Network::newImage(const ImageDesc& desc, ptrdiff_t offset)
  {
    assert(scratch);
    return scratch->newImage(desc, offset);
  }

  TensorDesc Network::getInputReorderDesc(const TensorDims& srcDims, int alignment)
  {
    assert(srcDims.size() == 3); // CHW

    TensorDims dstDims = srcDims;
    dstDims[0] = round_up(srcDims[0], blockSize); // round up C
    dstDims[1] = round_up(srcDims[1], int64_t(alignment)); // round up H
    dstDims[2] = round_up(srcDims[2], int64_t(alignment)); // round up W

    TensorLayout layout = blockSize == 16 ? TensorLayout::Chw16c : (blockSize == 8 ? TensorLayout::Chw8c : TensorLayout::chw);
    return TensorDesc(dstDims, layout, device->getTensorDataType());
  }

  std::shared_ptr<InputReorderNode> Network::addInputReorder(const std::string& name,
                                                             const std::shared_ptr<Tensor>& dst,
                                                             const std::shared_ptr<TransferFunction>& transferFunc,
                                                             bool hdr,
                                                             bool snorm)
  {
    auto node = device->newInputReorderNode({name, dst, transferFunc, hdr, snorm});
    nodes.push_back(node);
    return node;
  }

  std::shared_ptr<OutputReorderNode> Network::addOutputReorder(const std::string& name,
                                                               const std::shared_ptr<Tensor>& src,
                                                               const std::shared_ptr<TransferFunction>& transferFunc,
                                                               bool hdr,
                                                               bool snorm)
  {
    auto node = device->newOutputReorderNode({name, src, transferFunc, hdr, snorm});
    nodes.push_back(node);
    return node;
  }

  TensorDesc Network::getConvDesc(const std::string& name, const TensorDesc& srcDesc)
  {
    assert(srcDesc.ndims() == 3); // CHW

    const auto& bias = weightsMap[name + ".bias"];
    TensorDims dstDims = srcDesc.dims;
    dstDims[0] = round_up(bias->dims[0], blockSize); // dstDims[C] = round_up(OC, blockSize)
    return TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  std::shared_ptr<ConvNode> Network::addConv(const std::string& name,
                                             const std::shared_ptr<Tensor>& src,
                                             const std::shared_ptr<Tensor>& dst,
                                             bool relu)
  {
    assert(dst->desc() == getConvDesc(name, src->desc()));

    // Get and pad the weights
    auto weights = weightsMap[name + ".weight"];
    if (weights->ndims() != 4 || weights->layout != TensorLayout::oihw)
      throw Exception(Error::InvalidOperation, "invalid convolution weights");  
    if (blockSize > 1)
      weights = padWeights(weights);

    // Get and pad the biases
    auto bias = weightsMap[name + ".bias"];
    if (bias->ndims() != 1)
      throw Exception(Error::InvalidOperation, "invalid convolution biases");
    if (blockSize > 1)
      bias = padBias(bias);

    // Create the convolution node
    auto node = device->newConvNode({name, src, weights, bias, dst, relu});
    nodes.push_back(node);
    return node;
  }

  TensorDesc Network::getPoolDesc(const TensorDesc& srcDesc)
  {
    assert(srcDesc.ndims() == 3); // CHW

    TensorDims dstDims = srcDesc.dims;
    dstDims[1] /= 2; // H/2
    dstDims[2] /= 2; // W/2
    return TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  std::shared_ptr<PoolNode> Network::addPool(const std::string& name,
                                             const std::shared_ptr<Tensor>& src,
                                             const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->desc() == getPoolDesc(src->desc()));

    auto node = device->newPoolNode({name, src, dst});
    nodes.push_back(node);
    return node;
  }

  TensorDesc Network::getUpsampleDesc(const TensorDesc& srcDesc)
  {
    assert(srcDesc.ndims() == 3); // CHW

    TensorDims dstDims = srcDesc.dims;
    dstDims[1] *= 2; // H*2
    dstDims[2] *= 2; // W*2
    return TensorDesc(dstDims, srcDesc.layout, srcDesc.dataType);
  }

  std::shared_ptr<UpsampleNode> Network::addUpsample(const std::string& name,
                                                     const std::shared_ptr<Tensor>& src,
                                                     const std::shared_ptr<Tensor>& dst)
  {
    assert(dst->desc() == getUpsampleDesc(src->desc()));

    auto node = device->newUpsampleNode({name, src, dst});
    nodes.push_back(node);
    return node;
  }

  TensorDesc Network::getConcatDesc(const std::vector<TensorDesc>& srcDescs)
  {
    assert(!srcDescs.empty());
    assert(srcDescs[0].ndims() == 3); // CHW

    TensorDims dstDims = srcDescs[0].dims;
    for (size_t i = 1; i < srcDescs.size(); ++i)
    {
      assert(srcDescs[i].ndims() == 3); // CHW
      assert(srcDescs[i].dims[1] == srcDescs[0].dims[1]); // H
      assert(srcDescs[i].dims[2] == srcDescs[0].dims[2]); // W
      assert(srcDescs[i].layout == srcDescs[0].layout);
      assert(srcDescs[i].dataType == srcDescs[0].dataType);
      dstDims[0] += srcDescs[i].dims[0]; // C
    }
    return TensorDesc(dstDims, srcDescs[0].layout, srcDescs[0].dataType);
  }

  void Network::finalize()
  {
    // Compute the size of the scratch memory for the nodes
    size_t nodeScratchSize = 0;
    for (const auto& node : nodes)
      nodeScratchSize = max(nodeScratchSize, node->getScratchSize());

    // Allocate the scratch memory for the nodes
    TensorDims nodeScratchDims = { int64_t(nodeScratchSize) };
    auto nodeScratch = device->newTensor({nodeScratchDims, TensorLayout::x, DataType::UInt8});

    // Set the scratch memory for the nodes
    for (auto& node : nodes)
      node->setScratch(nodeScratch);

    // Free the weights
    weightsMap.clear();

    // Print statistics
    if (device->isVerbose(2))
    {
      const size_t scratchSize = scratch ? scratch->size() : 0;
      const size_t totalScratchSize = scratchSize + nodeScratchSize;
      std::cout << "Tensor scratch bytes: " << scratchSize << std::endl;
      std::cout << "Node scratch bytes  : " << nodeScratchSize << std::endl;
      std::cout << "Total scratch bytes : " << totalScratchSize << std::endl;
    }
  }

  std::shared_ptr<Tensor> Network::padWeights(const std::shared_ptr<Tensor>& src)
  {
    TensorAccessor4D<float, TensorLayout::oihw> srcAcc = *src;

    const int O1 = srcAcc.O;
    const int I1 = srcAcc.I;
    const int O2 = round_up(O1, blockSize);
    const int I2 = round_up(I1, blockSize);
    const int H = srcAcc.H;
    const int W = srcAcc.W;

    TensorDims dstDims = {O2, I2, H, W};
    //if (dstDims == src->dims)
    //  return src;

    auto dst = device->newTensor({dstDims, TensorLayout::oihw, DataType::Float32});
    TensorAccessor4D<float, TensorLayout::oihw> dstAcc = *dst;

    for (int o = 0; o < O2; ++o)
    {
      for (int i = 0; i < I2; ++i)
      {
        for (int h = 0; h < H; ++h)
        {
          for (int w = 0; w < W; ++w)
          {
            float value;
            if (o < O1 && i < I1)
              value = srcAcc(o, i, h, w);
            else
              value = 0; // padding

            dstAcc(o, i, h, w) = value;
          }
        }
      }
    }

    return dst;
  }

  std::shared_ptr<Tensor> Network::padBias(const std::shared_ptr<Tensor>& src)
  {
    TensorAccessor1D<float> srcAcc = *src;

    const int X1 = srcAcc.X;
    const int X2 = round_up(X1, blockSize);

    //if (X2 == X1)
    //  return src;

    auto dst = device->newTensor({TensorDims({X2}), TensorLayout::x, DataType::Float32});
    TensorAccessor1D<float> dstAcc = *dst;

    for (int x = 0; x < X1; ++x)
      dstAcc(x) = srcAcc(x);

    for (int x = X1; x < X2; ++x)
      dstAcc(x) = 0; // padding

    return dst;
  }

} // namespace oidn
