// ======================================================================== //
// Copyright 2009-2020 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "upsample.h"
#include "weights_reorder.h"
#include "autoexposure.h"
#include "network.h"

namespace oidn {

  template<int K>
  Network<K>::Network(const Ref<Device>& device, const std::map<std::string, Tensor>& weightMap)
    : device(device),
      eng(engine::kind::cpu, 0),
      sm(eng),
      weightMap(weightMap)
  {
  }

  template<int K>
  void Network<K>::execute(const Progress& progress, int taskIndex)
  {
    if (progress.func)
    {
      const double value = double(taskIndex) / double(progress.taskCount);
      if (!progress.func(progress.userPtr, value))
        throw Exception(Error::Cancelled, "execution was cancelled");
    }

    for (size_t i = 0; i < nodes.size(); ++i)
    {
      nodes[i]->execute(sm);

      if (progress.func)
      {
        const double value = (double(taskIndex) + double(i+1) / double(nodes.size())) / double(progress.taskCount);
        if (!progress.func(progress.userPtr, value))
          throw Exception(Error::Cancelled, "execution was cancelled");
      }
    }
  }

  template<int K>
  std::shared_ptr<memory> Network<K>::allocMemory(const memory::dims& dims,
                                                  memory::format_tag format,
                                                  void* data)
  {
    if (format == memory::format_tag::any)
    {
      if (dims.size() == 4)
        format = BlockedFormat<K>::nChwKc;
      else if (dims.size() == 1)
        format = memory::format_tag::x;
      else
        assert(0);
    }
    memory::desc desc(dims, memory::data_type::f32, format);
    if (data == nullptr)
    {
      const size_t bytes = getMemorySize(dims) * sizeof(float);
      if (format == BlockedFormat<K>::nChwKc)
        activationAllocBytes += bytes;
      totalAllocBytes += bytes;

      return std::make_shared<memory>(desc, eng);
    }
    else
    {
      return std::make_shared<memory>(desc, eng, data);
    }
  }

  template<int K>
  std::shared_ptr<memory> Network<K>::castMemory(const memory::dims& dims,
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
        format = BlockedFormat<K>::nChwKc;
      else if (dims.size() == 1)
        format = memory::format_tag::x;
      else
        assert(0);
    }
    memory::desc desc(dims, memory::data_type::f32, format);
    float* srcPtr = (float*)src->get_data_handle() + srcOffset;
    return std::make_shared<memory>(desc, eng, srcPtr);
  }

  template<int K>
  std::shared_ptr<memory> Network<K>::castMemory(const memory::dims& dims,
                                                 const std::shared_ptr<memory>& src,
                                                 const memory::dims& srcOffset)
  {
    return castMemory(dims, src, getMemorySize(srcOffset));
  }

  template<int K>
  void Network<K>::zeroMemory(const std::shared_ptr<memory>& dst)
  {
    assert(getMemoryType(dst) == memory::data_type::f32);
    memset(dst->get_data_handle(), 0, getMemorySize(dst)*sizeof(float));
  }

  template<int K>
  memory::dims Network<K>::getInputReorderDims(const memory::dims& srcDims, int alignment)
  {
    memory::dims dstDims = srcDims;
    dstDims[1] = round_up(srcDims[1], K); // round up C
    dstDims[2] = round_up(srcDims[2], memory::dim(alignment)); // round up H
    dstDims[3] = round_up(srcDims[3], memory::dim(alignment)); // round up W
    return dstDims;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::addInputReorder(const Image& color,
                                                    const Image& albedo,
                                                    const Image& normal,
                                                    const std::shared_ptr<TransferFunction>& transferFunc,
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
    auto node = std::make_shared<InputReorderNode>(color, albedo, normal, dst, transferFunc);
    nodes.push_back(node);
    return node;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::addOutputReorder(const std::shared_ptr<memory>& src,
                                                     const std::shared_ptr<TransferFunction>& transferFunc,
                                                     const Image& output)
  {
    memory::dims srcDims = getMemoryDims(src);
    assert(srcDims[1] == K);

    // Push node
    auto node = std::make_shared<OutputReorderNode>(src, output, transferFunc);
    nodes.push_back(node);
    return node;
  }

  template<int K>
  memory::dims Network<K>::getConvDims(const std::string& name, const memory::dims& srcDims)
  {
    auto b = weightMap[name + "/b"];
    memory::dims dstDims = srcDims;
    dstDims[1] = round_up(b.dims[0], K); // dstDims[C] = round_up(OC, K)
    return dstDims;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::addConv(const std::string& name,
                                            const std::shared_ptr<memory>& src,
                                            const std::shared_ptr<memory>& userDst,
                                            bool relu)
  {
    const memory::dims strides = {1, 1};
    const memory::dims padding = {1, 1};

    memory::dims srcDims = getMemoryDims(src);

    // Get the weights
    const auto& W = weightMap[name + "/W"];
    if (W.ndims() != 4 || W.format != "oihw")
      throw Exception(Error::InvalidOperation, "invalid convolution weights");
    memory::dims weightsDims = W.dims;
    auto userWeights = allocMemory(weightsDims, memory::format_tag::oihw, W.data);

    // Pad the weights
    memory::dims weightsPadDims = weightsDims;
    weightsPadDims[1] = round_up(weightsDims[1], K); // IC
    weightsPadDims[0] = round_up(weightsDims[0], K); // OC
    assert(srcDims[1] == weightsPadDims[1]); // srcDims[C] == weightsPadDims[IC]
    auto weightsPad = allocMemory(weightsPadDims, memory::format_tag::oihw);
    WeightsReorderNode<K>(userWeights, weightsPad).execute(sm);

    // Get the biases
    const auto& b = weightMap[name + "/b"];
    if (b.ndims() != 1)
      throw Exception(Error::InvalidOperation, "invalid convolution biases");
    memory::dims biasDims = b.dims;

    // Copy/pad the biases
    memory::dims biasPadDims = {round_up(biasDims[0], K)};
    auto bias = allocMemory(biasPadDims);
    if (biasDims[0] != biasPadDims[0])
      memset(bias->get_data_handle(), 0, biasPadDims[0]*sizeof(float));
    memcpy(bias->get_data_handle(), b.data, biasDims[0]*sizeof(float));

    // Allocate memory for destination
    memory::dims dstDims = srcDims;
    dstDims[1] = weightsPadDims[0]; // dstDims[C] = weightsPadDims[OC]

    std::shared_ptr<memory> dst;
    if (!userDst)
      dst = allocMemory(dstDims);
    else if (getMemoryDims(userDst) == dstDims)
      dst = userDst;
    else
      dst = castMemory(dstDims, userDst);

    // Create a convolution
    // Let the convolution primitive choose the weights format
    auto weightsDesc = memory::desc({ weightsPadDims }, memory::data_type::f32, memory::format_tag::any);

    auto convAlgo = (K == 16) ? algorithm::convolution_winograd : algorithm::convolution_direct;
    auto convDesc = convolution_forward::desc(
      prop_kind::forward_inference, convAlgo,
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
    auto weights = weightsPad;
    if (convPrimDesc.weights_desc() != weightsPad->get_desc())
    {
      weights = std::make_shared<memory>(convPrimDesc.weights_desc(), eng);
      ReorderNode(weightsPad, weights).execute(sm);
    }

    // Create convolution node and add it to the net
    auto node = std::make_shared<ConvNode>(convPrimDesc, src, weights, bias, dst);
    nodes.push_back(node);
    return node;
  }

  template<int K>
  memory::dims Network<K>::getPoolDims(const memory::dims& srcDims)
  {
    memory::dims dstDims = srcDims;
    dstDims[2] /= 2; // H/2
    dstDims[3] /= 2; // W/2
    return dstDims;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::addPool(const std::shared_ptr<memory>& src,
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

  template<int K>
  memory::dims Network<K>::getUpsampleDims(const memory::dims& srcDims)
  {
    memory::dims dstDims = srcDims;
    dstDims[2] *= 2; // H*2
    dstDims[3] *= 2; // W*2
    return dstDims;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::addUpsample(const std::shared_ptr<memory>& src,
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
    auto node = std::make_shared<UpsampleNode<K>>(src, dst);
    nodes.push_back(node);
    return node;
  }

  template<int K>
  memory::dims Network<K>::getConcatDims(const memory::dims& src1Dims, const memory::dims& src2Dims)
  {
    assert(src1Dims[0] == src2Dims[0]); // N
    assert(src1Dims[2] == src2Dims[2]); // H
    assert(src1Dims[3] == src2Dims[3]); // W

    memory::dims dstDims = src1Dims;
    dstDims[1] += src2Dims[1]; // C
    return dstDims;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::addAutoexposure(const Image& color,
                                                    const std::shared_ptr<TransferFunction>& transferFunc)
  {
    auto node = std::make_shared<AutoexposureNode>(color, transferFunc);
    nodes.push_back(node);
    return node;
  }

  template <int K>
  void Network<K>::finalize()
  {
    // Compute the size of the scratchpad
    size_t scratchpadSize = 0;
    for (const auto& node : nodes)
      scratchpadSize = max(scratchpadSize, node->getScratchpadSize());

    // Allocate the scratchpad
    memory::dims scratchpadDims = { memory::dim(scratchpadSize) };
    memory::desc scratchpadDesc(scratchpadDims, memory::data_type::u8, memory::format_tag::x);
    auto scratchpad = std::make_shared<memory>(scratchpadDesc, eng);
    activationAllocBytes += scratchpadSize;
    totalAllocBytes += scratchpadSize;

    // Set the scratchpad for the nodes
    for (auto& node : nodes)
      node->setScratchpad(scratchpad);

    // Free the weights
    weightMap.clear();

    // Print statistics
    if (device->isVerbose(2))
    {
      std::cout << "Activation bytes: " << activationAllocBytes << std::endl;
      std::cout << "Scratchpad bytes: " << scratchpadSize << std::endl;
      std::cout << "Total bytes     : " << totalAllocBytes << std::endl;
    }
  }

  template class Network<8>;
  template class Network<16>;

} // namespace oidn
