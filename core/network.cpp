// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "conv.h"
#include "pool.h"
#include "upsample.h"
#include "input_reorder.h"
#include "output_reorder.h"
#include "color.h"
#include "network.h"

namespace oidn {

  Network::Network(const Ref<Device>& device, const std::map<std::string, Ref<Tensor>>& weightsMap)
    : device(device),
      K(device->getTensorBlockSize()),
      weightsMap(weightsMap)
  {
  }

  void Network::execute(Progress& progress)
  {
    for (size_t i = 0; i < nodes.size(); ++i)
    {
      nodes[i]->execute();
      progress.update(1);
    }
  }

  double Network::getWorkAmount() const
  {
    return double(nodes.size());
  }

  Ref<Tensor> Network::newTensor(const TensorDims& dims)
  {
    assert(dims.size() == 3);
    
    TensorLayout layout;
    if (K == 16)
      layout = TensorLayout::Chw16c;
    else if (K == 8)
      layout = TensorLayout::Chw8c;
    else
      layout = TensorLayout::chw;
    
    Ref<Tensor> result = makeRef<Tensor>(device, dims, layout, DataType::Float32);
    
    size_t bytes = result->byteSize();
    activationAllocBytes += bytes;
    totalAllocBytes += bytes;

    return result;
  }

  TensorDims Network::getInputReorderDims(const TensorDims& srcDims, int alignment)
  {
    assert(srcDims.size() == 3); // CHW

    TensorDims dstDims = srcDims;
    dstDims[0] = round_up(srcDims[0], K); // round up C
    dstDims[1] = round_up(srcDims[1], int64_t(alignment)); // round up H
    dstDims[2] = round_up(srcDims[2], int64_t(alignment)); // round up W
    return dstDims;
  }

  Ref<Node> Network::addInputReorder(const Image& color,
                                     const Image& albedo,
                                     const Image& normal,
                                     const Ref<Tensor>& dst,
                                     const Ref<TransferFunction>& transferFunc,
                                     bool hdr,
                                     bool snorm,
                                     int alignment)
  {
    auto node = makeRef<InputReorderNode>(device, color, albedo, normal, dst, transferFunc, hdr, snorm);
    nodes.push_back(node);
    return node;
  }

  Ref<Node> Network::addOutputReorder(const Ref<Tensor>& src,
                                      const Image& output,
                                      const Ref<TransferFunction>& transferFunc,
                                      bool hdr,
                                      bool snorm)
  {
    auto node = makeRef<OutputReorderNode>(device, src, output, transferFunc, hdr, snorm);
    nodes.push_back(node);
    return node;
  }

  TensorDims Network::getConvDims(const std::string& name, const TensorDims& srcDims)
  {
    assert(srcDims.size() == 3); // CHW

    const auto& bias = weightsMap[name + ".bias"];
    TensorDims dstDims = srcDims;
    dstDims[0] = round_up(bias->dims[0], K); // dstDims[C] = round_up(OC, K)
    return dstDims;
  }

  Ref<Node> Network::addConv(const std::string& name,
                             const Ref<Tensor>& src,
                             const Ref<Tensor>& dst,
                             bool relu)
  {
    assert(dst->dims == getConvDims(name, src->dims));

    // Get and pad the weights
    auto weights = weightsMap[name + ".weight"];
    if (weights->ndims() != 4 || weights->layout != TensorLayout::oihw)
      throw Exception(Error::InvalidOperation, "invalid convolution weights");  
    if (K > 1)
      weights = padWeights(weights);

    // Get and pad the biases
    auto bias = weightsMap[name + ".bias"];
    if (bias->ndims() != 1)
      throw Exception(Error::InvalidOperation, "invalid convolution biases");
    if (K > 1)
      bias = padBias(bias);

    // Create the convolution node
    auto node = makeRef<ConvNode>(device, src, weights, bias, dst, relu);
    nodes.push_back(node);
    return node;
  }

  TensorDims Network::getPoolDims(const TensorDims& srcDims)
  {
    assert(srcDims.size() == 3); // CHW

    TensorDims dstDims = srcDims;
    dstDims[1] /= 2; // H/2
    dstDims[2] /= 2; // W/2
    return dstDims;
  }

  Ref<Node> Network::addPool(const Ref<Tensor>& src,
                             const Ref<Tensor>& dst)
  {
    assert(dst->dims == getPoolDims(src->dims));

    auto node = makeRef<PoolNode>(device, src, dst);
    nodes.push_back(node);
    return node;
  }

  TensorDims Network::getUpsampleDims(const TensorDims& srcDims)
  {
    assert(srcDims.size() == 3); // CHW

    TensorDims dstDims = srcDims;
    dstDims[1] *= 2; // H*2
    dstDims[2] *= 2; // W*2
    return dstDims;
  }

  Ref<Node> Network::addUpsample(const Ref<Tensor>& src,
                                 const Ref<Tensor>& dst)
  {
    assert(dst->dims == getUpsampleDims(src->dims));

    auto node = makeRef<UpsampleNode>(device, src, dst);
    nodes.push_back(node);
    return node;
  }

  TensorDims Network::getConcatDims(const std::vector<TensorDims>& srcDims)
  {
    assert(!srcDims.empty());
    assert(srcDims[0].size() == 3); // CHW

    TensorDims dstDims = srcDims[0];
    for (size_t i = 1; i < srcDims.size(); ++i)
    {
      assert(srcDims[i].size() == 3); // CHW
      assert(srcDims[i][1] == srcDims[0][1]); // H
      assert(srcDims[i][2] == srcDims[0][2]); // W
      dstDims[0] += srcDims[i][0]; // C
    }
    return dstDims;
  }

  std::vector<Ref<Tensor>> Network::getConcatSrc(const Ref<Tensor>& dst,
                                                 const std::vector<TensorDims>& srcDims)
  {
    assert(dst->dims == getConcatDims(srcDims));

    std::vector<Ref<Tensor>> src;
    size_t offset = 0;

    for (size_t i = 0; i < srcDims.size(); ++i)
    {
      src.push_back(dst->view(srcDims[i], offset));
      offset += getNumElements(srcDims[i]);
    }

    return src;
  }

  Ref<Node> Network::addAutoexposure(const Image& color,
                                     const Ref<TransferFunction>& transferFunc)
  {
    auto node = makeRef<AutoexposureNode>(device, color, transferFunc);
    nodes.push_back(node);
    return node;
  }

  void Network::finalize()
  {
    // Compute the size of the scratchpad
    size_t scratchpadSize = 0;
    for (const auto& node : nodes)
      scratchpadSize = max(scratchpadSize, node->getScratchpadSize());

    // Allocate the scratchpad
    TensorDims scratchpadDims = { int64_t(scratchpadSize) };
    auto scratchpad = makeRef<Tensor>(device, scratchpadDims, TensorLayout::x, DataType::UInt8);
    totalAllocBytes += scratchpadSize;

    // Set the scratchpad for the nodes
    for (auto& node : nodes)
      node->setScratchpad(scratchpad);

    // Free the weights
    weightsMap.clear();

    // Print statistics
    if (device->isVerbose(2))
    {
      std::cout << "Activation bytes: " << activationAllocBytes << std::endl;
      std::cout << "Scratchpad bytes: " << scratchpadSize << std::endl;
      std::cout << "Total bytes     : " << totalAllocBytes << std::endl;
    }
  }

  Ref<Tensor> Network::padWeights(const Ref<Tensor>& src)
  {
    assert(src->layout == TensorLayout::oihw);

    const int64_t O1 = src->dims[0];
    const int64_t I1 = src->dims[1];
    const int64_t O2 = round_up(O1, K);
    const int64_t I2 = round_up(I1, K);
    const int64_t H = src->dims[2];
    const int64_t W = src->dims[3];

    TensorDims dstDims = {O2, I2, H, W};
    if (dstDims == src->dims)
      return src;

    Ref<Tensor> dst = makeRef<Tensor>(device, dstDims, TensorLayout::oihw, DataType::Float32);

    for (int64_t o = 0; o < O2; ++o)
    {
      for (int64_t i = 0; i < I2; ++i)
      {
        for (int64_t h = 0; h < H; ++h)
        {
          for (int64_t w = 0; w < W; ++w)
          {
            float value;
            if (o < O1 && i < I1)
              value = src->get<float>(o, i, h, w);
            else
              value = 0; // padding

            dst->get<float>(o, i, h, w) = value;
          }
        }
      }
    }

    return dst;
  }

  Ref<Tensor> Network::padBias(const Ref<Tensor>& src)
  {
    assert(src->layout == TensorLayout::x);

    const int64_t X1 = src->dims[0];
    const int64_t X2 = round_up(X1, K);

    if (X2 == X1)
      return src;

    Ref<Tensor> dst = makeRef<Tensor>(device, TensorDims({X2}), TensorLayout::x, DataType::Float32);

    for (int64_t x = 0; x < X1; ++x)
      dst->get<float>(x) = src->get<float>(x);

    for (int64_t x = X1; x < X2; ++x)
      dst->get<float>(x) = 0; // padding

    return dst;
  }

} // namespace oidn
