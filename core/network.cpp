// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
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
#include "network.h"

namespace oidn {

  template<int K>
  Network<K>::Network(const std::map<std::string, Tensor>& weightMap)
    : cpuEngine(engine::cpu, 0),
      weightMap(weightMap)
  {
  }

  template<int K>
  void Network<K>::execute()
  {
    for (size_t i = 0; i < nodes.size(); ++i)
      nodes[i]->execute();
  }

  template<int K>
  std::shared_ptr<memory> Network<K>::allocTensor(const memory::dims& dims,
                                                  memory::format format,
                                                  void* data)
  {
    if (format == memory::format::any)
    {
      if (dims.size() == 4)
        format = BlockedFormat<K>::nChwKc;
      else if (dims.size() == 1)
        format = memory::format::x;
      else
        assert(0);
    }
    memory::desc desc(dims, memory::data_type::f32, format);
    memory::primitive_desc primDesc(desc, cpuEngine);
    if (data == nullptr)
      return std::make_shared<memory>(primDesc);
    else
      return std::make_shared<memory>(primDesc, data);
  }

  template<int K>
  std::shared_ptr<memory> Network<K>::castTensor(const memory::dims& dims,
                                                 const std::shared_ptr<memory>& src,
                                                 size_t srcOffset)
  {
    memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
    const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;
    MAYBE_UNUSED(srcDesc);
    assert(srcDesc.data_type == memory::data_type::f32);
    assert(srcDesc.format == BlockedFormat<K>::nChwKc);
    assert(dims[1] % K == 0); // C

    memory::desc desc(dims, memory::data_type::f32, BlockedFormat<K>::nChwKc);
    memory::primitive_desc primDesc(desc, cpuEngine);
    float* srcPtr = (float*)src->get_data_handle() + srcOffset;
    return std::make_shared<memory>(primDesc, srcPtr);
  }

  template<int K>
  std::shared_ptr<memory> Network<K>::castTensor(const memory::dims& dims,
                                                 const std::shared_ptr<memory>& src,
                                                 const memory::dims& srcOffset)
  {
    assert(srcOffset[1] % K == 0);   // C
    assert(srcOffset[2] == dims[2]); // H
    assert(srcOffset[3] == dims[3]); // W
    return castTensor(dims, src, getTensorSize(srcOffset));
  }

  template<int K>
  void Network<K>::zeroTensor(const std::shared_ptr<memory>& dst)
  {
    assert(getTensorType(dst) == memory::data_type::f32);
    memset(dst->get_data_handle(), 0, getTensorSize(dst)*sizeof(float));
  }

  template<int K>
  memory::dims Network<K>::getInputReorderDims(const memory::dims& srcDims, int spatialPad)
  {
    memory::dims dstDims = srcDims;
    dstDims[1] = getPadded<K>(srcDims[1]); // round up C
    dstDims[2] = (srcDims[2] + spatialPad - 1) / spatialPad * spatialPad; // round up H
    dstDims[3] = (srcDims[3] + spatialPad - 1) / spatialPad * spatialPad; // round up W
    return dstDims;
  }

  template<int K>
  memory::dims Network<K>::getConvDims(const std::string& name, const memory::dims& srcDims)
  {
    auto b = weightMap[name + "/b"];
    memory::dims dstDims = srcDims;
    dstDims[1] = getPadded<K>(b.dims[0]); // dstDims[C] = getPadded(OC)
    return dstDims;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::addConv(const std::string& name,
                                            const std::shared_ptr<memory>& src,
                                            bool relu)
  {
    const memory::dims strides = {1, 1};
    const memory::dims padding = {1, 1};

    memory::dims srcDims = getTensorDims(src);

    // Get the weights
    const auto& W = weightMap[name + "/W"];
    if (W.ndims() != 4 || W.format != "oihw")
      throw Exception(Error::InvalidOperation, "invalid convolution weights");
    memory::dims weightsDims = W.dims;
    auto userWeights = allocTensor(weightsDims, memory::format::oihw, W.data);

    // Pad the weights
    memory::dims weightsPadDims = weightsDims;
    weightsPadDims[1] = getPadded<K>(weightsDims[1]); // IC
    weightsPadDims[0] = getPadded<K>(weightsDims[0]); // OC
    assert(srcDims[1] == weightsPadDims[1]); // srcDims[C] == weightsPadDims[IC]
    auto weightsPad = allocTensor(weightsPadDims, memory::format::oihw);
    WeightsReorderNode<K>(userWeights, weightsPad).execute();

    // Get the biases
    const auto& b = weightMap[name + "/b"];
    if (b.ndims() != 1)
      throw Exception(Error::InvalidOperation, "invalid convolution biases");
    memory::dims biasDims = b.dims;

    // Copy/pad the biases
    memory::dims biasPadDims = {getPadded<K>(biasDims[0])};
    auto bias = allocTensor(biasPadDims);
    if (biasDims[0] != biasPadDims[0])
      memset(bias->get_data_handle(), 0, biasPadDims[0]*sizeof(float));
    memcpy(bias->get_data_handle(), b.data, biasDims[0]*sizeof(float));

    // Allocate memory for destination
    memory::dims dstDims = srcDims;
    dstDims[1] = weightsPadDims[0]; // dstDims[C] = weightsPadDims[OC]
    auto dst = allocTensor(dstDims);
    assert(getTensorDims(dst) == dstDims);

    // Create a convolution
    // Let the convolution primitive choose the weights format
    auto weightsDesc = memory::desc({ weightsPadDims }, memory::data_type::f32, memory::format::any);

    auto convAlgo = (K == 16) ? convolution_winograd : convolution_direct;
    auto convDesc = convolution_forward::desc(
      prop_kind::forward_inference, convAlgo,
      src->get_primitive_desc().desc(),
      weightsDesc,
      bias->get_primitive_desc().desc(),
      dst->get_primitive_desc().desc(),
      strides, padding, padding, padding_kind::zero);

    // Incorporate relu
    mkldnn::primitive_attr convAttr;
    if (relu)
    {
      mkldnn::post_ops ops;
      ops.append_eltwise(
        1.f,   // scale factor, not used
        algorithm::eltwise_relu,
        0.f,   // max with
        0.f    // unused
      );
      convAttr.set_post_ops(ops);
    }

    auto convPrimDesc = convolution_forward::primitive_desc(convDesc, convAttr, cpuEngine);

    // Reorder the weights to the final format, if necessary
    auto weights = weightsPad;
    if (memory::primitive_desc(convPrimDesc.weights_primitive_desc()) != weightsPad->get_primitive_desc())
    {
      weights = std::make_shared<memory>(convPrimDesc.weights_primitive_desc());
      MklNode(reorder(*weightsPad, *weights)).execute();
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

    memory::dims srcDims = getTensorDims(src);
    memory::dims dstDims = getPoolDims(srcDims);

    auto dst = userDst;
    if (!dst)
      dst = allocTensor(dstDims);
    assert(getTensorDims(dst) == dstDims);

    auto poolDesc = pooling_forward::desc(
      prop_kind::forward_inference, pooling_max,
      src->get_primitive_desc().desc(),
      dst->get_primitive_desc().desc(),
      strides, kernel, padding, padding, padding_kind::zero);
    auto poolPrimDesc = pooling_forward::primitive_desc(poolDesc, cpuEngine);

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
    memory::dims srcDims = getTensorDims(src);
    memory::dims dstDims = getUpsampleDims(srcDims);

    auto dst = userDst;
    if (!dst)
      dst = allocTensor(dstDims);
    assert(getTensorDims(dst) == dstDims);

    // Create upsampling node and add it to net
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

  template class Network<8>;
  template class Network<16>;

} // namespace oidn
