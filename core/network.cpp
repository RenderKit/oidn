// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "upsample.h"
#include "color.h"
#include "network.h"

namespace oidn {

  Network::Network(const Ref<Device>& device, const std::map<std::string, Ref<Tensor>>& weightsMap)
    : device(device),
      weightsMap(weightsMap)
  {
    // Determine the block size for blocked tensor layouts
    K = mayiuse(avx512_core) ? 16 : 8;
  }

  void Network::execute(Progress& progress)
  {
    for (size_t i = 0; i < nodes.size(); ++i)
    {
      nodes[i]->execute(device->getStream());
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
    
    TensorLayout layout = (K == 16) ? TensorLayout::Chw16c : TensorLayout::Chw8c;
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

  std::shared_ptr<Node> Network::addInputReorder(const Image& color,
                                                 const Image& albedo,
                                                 const Image& normal,
                                                 const std::shared_ptr<TransferFunction>& transferFunc,
                                                 bool hdr,
                                                 int alignment,
                                                 const Ref<Tensor>& dst)
  {
    assert(color);
    int inputC = 3;
    if (albedo) inputC += 3;
    if (normal) inputC += 3;
    assert(dst->dims == getInputReorderDims({inputC, color.height, color.width}, alignment));

    // Push node
    auto node = std::make_shared<InputReorderNode>(color, albedo, normal, dst, transferFunc, hdr);
    nodes.push_back(node);
    return node;
  }

  std::shared_ptr<Node> Network::addOutputReorder(const Ref<Tensor>& src,
                                                  const std::shared_ptr<TransferFunction>& transferFunc,
                                                  bool hdr,
                                                  const Image& output)
  {
    assert(src->ndims() == 3); // CHW
    assert(src->dims[0] == K);

    // Push node
    auto node = std::make_shared<OutputReorderNode>(src, output, transferFunc, hdr);
    nodes.push_back(node);
    return node;
  }

  TensorDims Network::getConvDims(const std::string& name, const TensorDims& srcDims)
  {
    assert(srcDims.size() == 3); // CHW

    const auto& b = weightsMap[name + ".bias"];
    TensorDims dstDims = srcDims;
    dstDims[0] = round_up(b->dims[0], K); // dstDims[C] = round_up(OC, K)
    return dstDims;
  }

  std::shared_ptr<Node> Network::addConv(const std::string& name,
                                         const Ref<Tensor>& src,
                                         const Ref<Tensor>& dst,
                                         bool relu)
  {
    assert(dst->dims == getConvDims(name, src->dims));

    const dnnl::memory::dims strides = {1, 1};
    const dnnl::memory::dims padding = {1, 1};

    // Get and pad the weights
    const auto& W = weightsMap[name + ".weight"];
    if (W->ndims() != 4 || W->layout != TensorLayout::oihw)
      throw Exception(Error::InvalidOperation, "invalid convolution weights");
    auto weights = padWeights(W);
    TensorDims weightsDims = weights->dims;

    // Get and pad the biases
    const auto& b = weightsMap[name + ".bias"];
    if (b->ndims() != 1)
      throw Exception(Error::InvalidOperation, "invalid convolution biases");
    auto bias = padBias(b);
    TensorDims biasDims = bias->dims;

    // Create a convolution
    // Let the convolution primitive choose the weights format
    auto weightsDesc = dnnl::memory::desc({ weightsDims },
                                          dnnl::memory::data_type::f32,
                                          dnnl::memory::format_tag::any);

    auto convDesc = dnnl::convolution_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
      src->mem.get_desc(),
      weightsDesc,
      bias->mem.get_desc(),
      dst->mem.get_desc(),
      strides, padding, padding);

    // Incorporate relu
    dnnl::primitive_attr convAttr;
    if (relu)
    {
      dnnl::post_ops ops;
      ops.append_eltwise(
        1.f,   // scale
        dnnl::algorithm::eltwise_relu,
        0.f,   // alpha
        0.f    // beta
      );
      convAttr.set_post_ops(ops);
    }
    convAttr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto convPrimDesc = dnnl::convolution_forward::primitive_desc(convDesc, convAttr, device->getEngine());

    // Reorder the weights to the final format, if necessary
    if (convPrimDesc.weights_desc() != weights->mem.get_desc())
    {
      auto oldWeights = weights;
      weights = makeRef<Tensor>(device, convPrimDesc.weights_desc());
      ReorderNode(oldWeights, weights).execute(device->getStream());
    }

    // Create convolution node and add it to the net
    auto node = std::make_shared<ConvNode>(convPrimDesc, src, weights, bias, dst);
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

  std::shared_ptr<Node> Network::addPool(const Ref<Tensor>& src,
                                         const Ref<Tensor>& dst)
  {
    assert(dst->dims == getPoolDims(src->dims));

    const dnnl::memory::dims kernel  = {2, 2};
    const dnnl::memory::dims strides = {2, 2};
    const dnnl::memory::dims padding = {0, 0};

    auto poolDesc = dnnl::pooling_forward::desc(
      dnnl::prop_kind::forward_inference, dnnl::algorithm::pooling_max,
      src->mem.get_desc(),
      dst->mem.get_desc(),
      strides, kernel, padding, padding);

    dnnl::primitive_attr poolAttr;
    poolAttr.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto poolPrimDesc = dnnl::pooling_forward::primitive_desc(poolDesc, poolAttr, device->getEngine());
    auto node = std::make_shared<PoolNode>(poolPrimDesc, src, dst);
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

  std::shared_ptr<Node> Network::addUpsample(const Ref<Tensor>& src,
                                             const Ref<Tensor>& dst)
  {
    assert(dst->dims == getUpsampleDims(src->dims));

    // Create upsampling node and add it to net
    /*
    auto resampleDesc = resampling_forward::desc(
      prop_kind::forward_inference, algorithm::resampling_nearest,
      src->get_desc(),
      dst->get_desc());

    dnnl::primitive_attr resampleAttr;
    resampleAttr.set_scratchpad_mode(scratchpad_mode::user);

    auto resamplePrimDesc = resampling_forward::primitive_desc(resampleDesc, resampleAttr, device->getEngine());
    auto node = std::make_shared<ResampleNode>(resamplePrimDesc, src, dst);
    */
    auto node = std::make_shared<UpsampleNode>(K, src, dst);
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

  std::vector<Ref<Tensor>> Network::getConcatSrc(const Ref<Tensor>& dst, const std::vector<TensorDims>& srcDims)
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

  std::shared_ptr<Node> Network::addAutoexposure(const Image& color,
                                                 const std::shared_ptr<TransferFunction>& transferFunc)
  {
    auto node = std::make_shared<AutoexposureNode>(color, transferFunc);
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
