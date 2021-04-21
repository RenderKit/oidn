// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "upsample.h"
#include "color.h"
#include "network.h"

namespace oidn {

  Network::Network(const Ref<Device>& device, const std::map<std::string, Tensor>& weightsMap)
    : device(device),
      eng(engine::kind::cpu, 0),
      sm(eng),
      weightsMap(weightsMap)
  {
    if (mayiuse(avx512_core))
    {
      K = 16;
      nChwKc = memory::format_tag::nChw16c;
    }
    else
    {
      K = 8;
      nChwKc = memory::format_tag::nChw8c;
    }
  }

  void Network::execute(Progress& progress)
  {
    for (size_t i = 0; i < nodes.size(); ++i)
    {
      nodes[i]->execute(sm);
      progress.update(1);
    }
  }

  double Network::getWorkAmount() const
  {
    return double(nodes.size());
  }

  std::shared_ptr<memory> Network::allocMemory(const memory::dims& dims,
                                               memory::format_tag format,
                                               void* data)
  {
    if (format == memory::format_tag::any)
    {
      if (dims.size() == 4)
        format = nChwKc;
      else if (dims.size() == 1)
        format = memory::format_tag::x;
      else
        assert(0);
    }
    memory::desc desc(dims, memory::data_type::f32, format);
    if (data == nullptr)
    {
      const size_t bytes = getMemorySize(dims) * sizeof(float);
      if (format == nChwKc)
        activationAllocBytes += bytes;
      totalAllocBytes += bytes;

      return std::make_shared<memory>(desc, eng);
    }
    else
    {
      return std::make_shared<memory>(desc, eng, data);
    }
  }

  std::shared_ptr<memory> Network::castMemory(const memory::dims& dims,
                                              const std::shared_ptr<memory>& src,
                                              size_t srcOffset,
                                              memory::format_tag format)
  {
    const dnnl_memory_desc_t& srcDesc = src->get_desc().data;
    MAYBE_UNUSED(srcDesc);
    assert(srcDesc.data_type == memory::data_type::f32);
    assert(getMemorySize(src) >= srcOffset + getMemorySize(dims));

    if (format == memory::format_tag::any)
    {
      if (dims.size() == 4)
        format = nChwKc;
      else if (dims.size() == 1)
        format = memory::format_tag::x;
      else
        assert(0);
    }
    memory::desc desc(dims, memory::data_type::f32, format);
    float* srcPtr = (float*)src->get_data_handle() + srcOffset;
    return std::make_shared<memory>(desc, eng, srcPtr);
  }

  std::shared_ptr<memory> Network::castMemory(const memory::dims& dims,
                                              const std::shared_ptr<memory>& src,
                                              const memory::dims& srcOffset)
  {
    return castMemory(dims, src, getMemorySize(srcOffset));
  }

  void Network::zeroMemory(const std::shared_ptr<memory>& dst)
  {
    assert(getMemoryType(dst) == memory::data_type::f32);
    memset(dst->get_data_handle(), 0, getMemorySize(dst)*sizeof(float));
  }

  memory::dims Network::getInputReorderDims(const memory::dims& srcDims, int alignment)
  {
    memory::dims dstDims = srcDims;
    dstDims[1] = round_up(srcDims[1], K); // round up C
    dstDims[2] = round_up(srcDims[2], memory::dim(alignment)); // round up H
    dstDims[3] = round_up(srcDims[3], memory::dim(alignment)); // round up W
    return dstDims;
  }

  std::shared_ptr<Node> Network::addInputReorder(const Image& color,
                                                 const Image& albedo,
                                                 const Image& normal,
                                                 const std::shared_ptr<TransferFunction>& transferFunc,
                                                 bool hdr,
                                                 int alignment,
                                                 const std::shared_ptr<memory>& userDst)
  {
    assert(color);
    int inputC = 3;
    if (albedo) inputC += 3;
    if (normal) inputC += 3;

    memory::dims srcDims = {1, inputC, color.height, color.width};
    memory::dims dstDims = getInputReorderDims(srcDims, alignment);

    // Allocate padded memory
    auto dst = userDst;
    if (!dst)
      dst = allocMemory(dstDims);

    // Push node
    auto node = std::make_shared<InputReorderNode>(color, albedo, normal, dst, transferFunc, hdr);
    nodes.push_back(node);
    return node;
  }

  std::shared_ptr<Node> Network::addOutputReorder(const std::shared_ptr<memory>& src,
                                                  const std::shared_ptr<TransferFunction>& transferFunc,
                                                  bool hdr,
                                                  const Image& output)
  {
    memory::dims srcDims = getMemoryDims(src);
    assert(srcDims[1] == K);

    // Push node
    auto node = std::make_shared<OutputReorderNode>(src, output, transferFunc, hdr);
    nodes.push_back(node);
    return node;
  }

  memory::dims Network::getConvDims(const std::string& name, const memory::dims& srcDims)
  {
    auto b = weightsMap[name + ".bias"];
    memory::dims dstDims = srcDims;
    dstDims[1] = round_up(b.dims[0], K); // dstDims[C] = round_up(OC, K)
    return dstDims;
  }

  std::shared_ptr<Node> Network::addConv(const std::string& name,
                                         const std::shared_ptr<memory>& src,
                                         const std::shared_ptr<memory>& userDst,
                                         bool relu)
  {
    const memory::dims strides = {1, 1};
    const memory::dims padding = {1, 1};

    memory::dims srcDims = getMemoryDims(src);

    // Get and pad the weights
    const auto& W = weightsMap[name + ".weight"];
    if (W.ndims() != 4 || W.layout != "oihw")
      throw Exception(Error::InvalidOperation, "invalid convolution weights");
    auto weights = padWeights(W);
    memory::dims weightsDims = getMemoryDims(weights);

    // Get and pad the biases
    const auto& b = weightsMap[name + ".bias"];
    if (b.ndims() != 1)
      throw Exception(Error::InvalidOperation, "invalid convolution biases");
    auto bias = padBias(b);
    memory::dims biasDims = getMemoryDims(bias);

    // Allocate memory for destination
    memory::dims dstDims = srcDims;
    dstDims[1] = weightsDims[0]; // dstDims[C] = weightsDims[OC]

    std::shared_ptr<memory> dst;
    if (!userDst)
      dst = allocMemory(dstDims);
    else if (getMemoryDims(userDst) == dstDims)
      dst = userDst;
    else
      dst = castMemory(dstDims, userDst);

    // Create a convolution
    // Let the convolution primitive choose the weights format
    auto weightsDesc = memory::desc({ weightsDims }, memory::data_type::f32, memory::format_tag::any);

    auto convDesc = convolution_forward::desc(
      prop_kind::forward_inference, algorithm::convolution_direct,
      src->get_desc(),
      weightsDesc,
      bias->get_desc(),
      dst->get_desc(),
      strides, padding, padding);

    // Incorporate relu
    dnnl::primitive_attr convAttr;
    if (relu)
    {
      dnnl::post_ops ops;
      ops.append_eltwise(
        1.f,   // scale
        algorithm::eltwise_relu,
        0.f,   // alpha
        0.f    // beta
      );
      convAttr.set_post_ops(ops);
    }
    convAttr.set_scratchpad_mode(scratchpad_mode::user);

    auto convPrimDesc = convolution_forward::primitive_desc(convDesc, convAttr, eng);

    // Reorder the weights to the final format, if necessary
    if (convPrimDesc.weights_desc() != weights->get_desc())
    {
      auto oldWeights = weights;
      weights = std::make_shared<memory>(convPrimDesc.weights_desc(), eng);
      ReorderNode(oldWeights, weights).execute(sm);
    }

    // Create convolution node and add it to the net
    auto node = std::make_shared<ConvNode>(convPrimDesc, src, weights, bias, dst);
    nodes.push_back(node);
    return node;
  }

  memory::dims Network::getPoolDims(const memory::dims& srcDims)
  {
    memory::dims dstDims = srcDims;
    dstDims[2] /= 2; // H/2
    dstDims[3] /= 2; // W/2
    return dstDims;
  }

  std::shared_ptr<Node> Network::addPool(const std::shared_ptr<memory>& src,
                                         const std::shared_ptr<memory>& userDst)
  {
    const memory::dims kernel  = {2, 2};
    const memory::dims strides = {2, 2};
    const memory::dims padding = {0, 0};

    memory::dims srcDims = getMemoryDims(src);
    memory::dims dstDims = getPoolDims(srcDims);

    std::shared_ptr<memory> dst;
    if (!userDst)
      dst = allocMemory(dstDims);
    else if (getMemoryDims(userDst) == dstDims)
      dst = userDst;
    else
      dst = castMemory(dstDims, userDst);

    auto poolDesc = pooling_forward::desc(
      prop_kind::forward_inference, algorithm::pooling_max,
      src->get_desc(),
      dst->get_desc(),
      strides, kernel, padding, padding);

    dnnl::primitive_attr poolAttr;
    poolAttr.set_scratchpad_mode(scratchpad_mode::user);

    auto poolPrimDesc = pooling_forward::primitive_desc(poolDesc, poolAttr, eng);
    auto node = std::make_shared<PoolNode>(poolPrimDesc, src, dst);
    nodes.push_back(node);
    return node;
  }

  memory::dims Network::getUpsampleDims(const memory::dims& srcDims)
  {
    memory::dims dstDims = srcDims;
    dstDims[2] *= 2; // H*2
    dstDims[3] *= 2; // W*2
    return dstDims;
  }

  std::shared_ptr<Node> Network::addUpsample(const std::shared_ptr<memory>& src,
                                             const std::shared_ptr<memory>& userDst)
  {
    memory::dims srcDims = getMemoryDims(src);
    memory::dims dstDims = getUpsampleDims(srcDims);

    std::shared_ptr<memory> dst;
    if (!userDst)
      dst = allocMemory(dstDims);
    else if (getMemoryDims(userDst) == dstDims)
      dst = userDst;
    else
      dst = castMemory(dstDims, userDst);

    // Create upsampling node and add it to net
    /*
    auto resampleDesc = resampling_forward::desc(
      prop_kind::forward_inference, algorithm::resampling_nearest,
      src->get_desc(),
      dst->get_desc());

    dnnl::primitive_attr resampleAttr;
    resampleAttr.set_scratchpad_mode(scratchpad_mode::user);

    auto resamplePrimDesc = resampling_forward::primitive_desc(resampleDesc, resampleAttr, eng);
    auto node = std::make_shared<ResampleNode>(resamplePrimDesc, src, dst);
    */
    auto node = std::make_shared<UpsampleNode>(K, src, dst);
    nodes.push_back(node);
    return node;
  }

  memory::dims Network::getConcatDims(const memory::dims& src1Dims, const memory::dims& src2Dims)
  {
    assert(src1Dims[0] == src2Dims[0]); // N
    assert(src1Dims[2] == src2Dims[2]); // H
    assert(src1Dims[3] == src2Dims[3]); // W

    memory::dims dstDims = src1Dims;
    dstDims[1] += src2Dims[1]; // C
    return dstDims;
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
    memory::dims scratchpadDims = { memory::dim(scratchpadSize) };
    memory::desc scratchpadDesc(scratchpadDims, memory::data_type::u8, memory::format_tag::x);
    auto scratchpad = std::make_shared<memory>(scratchpadDesc, eng);
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

  std::shared_ptr<memory> Network::padWeights(const Tensor& src)
  {
    assert(src.layout == "oihw");

    const int64_t O1 = src.dims[0];
    const int64_t I1 = src.dims[1];
    const int64_t O2 = round_up(O1, K);
    const int64_t I2 = round_up(I1, K);
    const int64_t H = src.dims[2];
    const int64_t W = src.dims[3];

    Tensor::Dims dstDims = {O2, I2, H, W};
    if (dstDims == src.dims)
      return allocMemory(src.dims, memory::format_tag::oihw, src.data);

    std::shared_ptr<memory> dstMem = allocMemory(dstDims, memory::format_tag::oihw);
    Tensor dst(dstDims, src.layout, (float*)dstMem->map_data());

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
              value = src(o, i, h, w);
            else
              value = 0; // padding;

            dst(o, i, h, w) = value;
          }
        }
      }
    }

    dstMem->unmap_data(dst.data);
    return dstMem;
  }

  std::shared_ptr<memory> Network::padBias(const Tensor& src)
  {
    assert(src.ndims() == 1);

    const int64_t X1 = src.dims[0];
    const int64_t X2 = round_up(X1, K);

    if (X2 == X1)
      return allocMemory(src.dims, memory::format_tag::x, src.data);

    std::shared_ptr<memory> dstMem = allocMemory({X2}, memory::format_tag::x);
    Tensor dst({X2}, src.layout, (float*)dstMem->map_data());

    for (int64_t x = 0; x < X1; ++x)
      dst(x) = src(x);

    for (int64_t x = X1; x < X2; ++x)
      dst(x) = 0; // padding

    dstMem->unmap_data(dst.data);
    return dstMem;
  }

} // namespace oidn
