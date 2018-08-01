// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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
#include "input_reorder.h"
#include "output_reorder.h"
#include "network.h"

namespace oidn {

  template<int K>
  Network<K>::Network(const std::map<std::string, Tensor>& weight_map)
    : cpu_engine(engine::cpu, 0),
      weight_map(weight_map)
  {
  }

  template<int K>
  void Network<K>::execute()
  {
    for (size_t i = 0; i < nodes.size(); ++i)
      nodes[i]->execute();
  }

  template<int K>
  std::shared_ptr<memory> Network<K>::alloc_tensor(const memory::dims& dims,
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
    memory::desc md(dims, memory::data_type::f32, format);
    memory::primitive_desc mpd(md, cpu_engine);
    if (data == nullptr)
      return std::make_shared<memory>(mpd);
    else
      return std::make_shared<memory>(mpd, data);
  }

  template<int K>
  std::shared_ptr<memory> Network<K>::cast_tensor(const memory::dims& dims,
                                                  const std::shared_ptr<memory>& src,
                                                  size_t src_offset)
  {
    memory::primitive_desc src_mpd = src->get_primitive_desc();
    const mkldnn_memory_desc_t& src_md = src_mpd.desc().data;
    MAYBE_UNUSED(src_md);
    assert(src_md.data_type == memory::data_type::f32);
    assert(src_md.format == BlockedFormat<K>::nChwKc);
    assert(dims[1] % K == 0); // C

    memory::desc md(dims, memory::data_type::f32, BlockedFormat<K>::nChwKc);
    memory::primitive_desc mpd(md, cpu_engine);
    float* src_ptr = (float*)src->get_data_handle() + src_offset;
    return std::make_shared<memory>(mpd, src_ptr);
  }

  template<int K>
  std::shared_ptr<memory> Network<K>::cast_tensor(const memory::dims& dims,
                                                  const std::shared_ptr<memory>& src,
                                                  const memory::dims& src_offset)
  {
    assert(src_offset[1] % K == 0);   // C
    assert(src_offset[2] == dims[2]); // H
    assert(src_offset[3] == dims[3]); // W
    return cast_tensor(dims, src, tensor_size(src_offset));
  }

  template<int K>
  void Network<K>::zero_tensor(const std::shared_ptr<memory>& dst)
  {
    assert(tensor_type(dst) == memory::data_type::f32);
    memset(dst->get_data_handle(), 0, tensor_size(dst)*sizeof(float));
  }

  template<int K>
  void Network<K>::weights_reorder(const std::shared_ptr<memory>& src,
                                   const std::shared_ptr<memory>& dst)
  {
    WeightsReorder<K>(src, dst).execute();
  }

  template<int K>
  memory::dims Network<K>::input_reorder_dims(const memory::dims& src_tz, int spatial_pad)
  {
    memory::dims dst_tz = src_tz;
    dst_tz[1] = padded<K>(src_tz[1]); // round up C
    dst_tz[2] = (src_tz[2] + spatial_pad - 1) / spatial_pad * spatial_pad; // round up H
    dst_tz[3] = (src_tz[3] + spatial_pad - 1) / spatial_pad * spatial_pad; // round up W
    return dst_tz;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::add_input_reorder(const BufferView2D& src,
                                                      const BufferView2D& src_albedo,
                                                      const BufferView2D& src_normal,
                                                      int spatial_pad,
                                                      const std::shared_ptr<memory>& user_dst)
  {
    memory::dims src_tz = {1, 9, src.height, src.width};
    memory::dims dst_tz = input_reorder_dims(src_tz, spatial_pad);

    // Allocate padded memory
    auto dst = user_dst;
    if (!dst)
      dst = alloc_tensor(dst_tz);
    assert(tensor_dims(dst) == dst_tz);

    // Push node
    std::shared_ptr<Node> node;
    if (src.format == Format::FLOAT3) // need convert to sRGB?
      node = std::make_shared<InputReorder<K, true>>(src, src_albedo, src_normal, dst);
    else if (src.format == Format::FLOAT3_SRGB) // no conversion?
      node = std::make_shared<InputReorder<K, false>>(src, src_albedo, src_normal, dst);
    else
      assert(0);

    nodes.push_back(node);
    return node;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::add_output_reorder(const std::shared_ptr<memory>& src,
                                                       const BufferView2D& dst)
  {
    memory::dims src_tz = tensor_dims(src);
    assert(src_tz[1] == K);

    // Push node
    std::shared_ptr<Node> node;
    if (dst.format == Format::FLOAT3) // need convert to linear?
      node = std::make_shared<OutputReorder<K, true>>(src, dst);
    else if (dst.format == Format::FLOAT3_SRGB) // no conversion?
      node = std::make_shared<OutputReorder<K, false>>(src, dst);
    else
      assert(0);

    nodes.push_back(node);
    return node;
  }

  template<int K>
  memory::dims Network<K>::conv_dims(const std::string& name, const memory::dims& src_tz)
  {
    auto b = weight_map[name + "/b"];
    memory::dims dst_tz = src_tz;
    dst_tz[1] = padded<K>(b.dims[0]); // dst_tz[C] = padded(OC)
    return dst_tz;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::add_conv(const std::string& name,
                                             const std::shared_ptr<memory>& src,
                                             bool relu)
  {
    const memory::dims strides = {1, 1};
    const memory::dims padding = {1, 1};

    memory::dims src_tz = tensor_dims(src);

    // Get the weights
    const auto& W = weight_map[name + "/W"];
    if (W.ndims() != 4 || W.format != "oihw")
      throw std::runtime_error("invalid weights");
    memory::dims weights_tz = W.dims;
    auto user_weights = alloc_tensor(weights_tz, memory::format::oihw, W.data);

    // Reorder/pad the weights
    memory::dims weights_pad_tz = weights_tz;
    weights_pad_tz[1] = padded<K>(weights_tz[1]); // IC
    weights_pad_tz[0] = padded<K>(weights_tz[0]); // OC
    assert(src_tz[1] == weights_pad_tz[1]); // src_tz[C] == weights_pad_tz[IC]
    auto weights = alloc_tensor(weights_pad_tz, BlockedFormat<K>::OIhwKiKo);
    weights_reorder(user_weights, weights);

    // Get the biases
    const auto& b = weight_map[name + "/b"];
    if (b.ndims() != 1)
      throw std::runtime_error("invalid biases");
    memory::dims bias_tz = b.dims;

    // Copy/pad the biases
    memory::dims bias_pad_tz = {padded<K>(bias_tz[0])};
    auto bias = alloc_tensor(bias_pad_tz);
    if (bias_tz[0] != bias_pad_tz[0])
      memset(bias->get_data_handle(), 0, bias_pad_tz[0]*sizeof(float));
    memcpy(bias->get_data_handle(), b.data, bias_tz[0]*sizeof(float));

    // Allocate memory for destination
    memory::dims dst_tz = src_tz;
    dst_tz[1] = weights_pad_tz[0]; // dst_tz[C] = weights_pad_tz[OC]
    auto dst = alloc_tensor(dst_tz);
    assert(tensor_dims(dst) == dst_tz);

    // Create a convolution
    auto conv_algo = (K == 16) ? convolution_winograd : convolution_direct;
    auto conv_desc = convolution_forward::desc(
      prop_kind::forward_inference, conv_algo,
      src->get_primitive_desc().desc(),
      weights->get_primitive_desc().desc(),
      bias->get_primitive_desc().desc(),
      dst->get_primitive_desc().desc(),
      strides, padding, padding, padding_kind::zero);

    // Incorporate relu
    mkldnn::primitive_attr conv_attr;
    if (relu)
    {
      mkldnn::post_ops ops;
      ops.append_eltwise(
        1.f,   // scale factor, not used
        algorithm::eltwise_relu,
        0.f,   // max with
        0.f    // unused
      );
      conv_attr.set_post_ops(ops);
    }

    auto conv_pd = convolution_forward::primitive_desc(conv_desc, conv_attr, cpu_engine);

    // Create convolution node and add it to the net
    auto node = std::make_shared<Conv>(conv_pd, src, weights, bias, dst);
    nodes.push_back(node);
    return node;
  }

  template<int K>
  memory::dims Network<K>::pool_dims(const memory::dims& src_tz)
  {
    memory::dims dst_tz = src_tz;
    dst_tz[2] /= 2; // H/2
    dst_tz[3] /= 2; // W/2
    return dst_tz;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::add_pool(const std::shared_ptr<memory>& src,
                                             const std::shared_ptr<memory>& user_dst)
  {
    const memory::dims kernel  = {2, 2};
    const memory::dims strides = {2, 2};
    const memory::dims padding = {0, 0};

    memory::dims src_tz = tensor_dims(src);
    memory::dims dst_tz = pool_dims(src_tz);

    auto dst = user_dst;
    if (!dst)
      dst = alloc_tensor(dst_tz);
    assert(tensor_dims(dst) == dst_tz);

    auto pool_desc = pooling_forward::desc(
      prop_kind::forward_inference, pooling_max,
      src->get_primitive_desc().desc(),
      dst->get_primitive_desc().desc(),
      strides, kernel, padding, padding, padding_kind::zero);
    auto pool_pd = pooling_forward::primitive_desc(pool_desc, cpu_engine);

    auto node = std::make_shared<Pool>(pool_pd, src, dst);
    nodes.push_back(node);
    return node;
  }

  template<int K>
  memory::dims Network<K>::unpool_dims(const memory::dims& src_tz)
  {
    memory::dims dst_tz = src_tz;
    dst_tz[2] *= 2; // H*2
    dst_tz[3] *= 2; // W*2
    return dst_tz;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::add_unpool(const std::shared_ptr<memory>& src,
                                               const std::shared_ptr<memory>& user_dst)
  {
    memory::dims src_tz = tensor_dims(src);
    memory::dims dst_tz = unpool_dims(src_tz);

    auto dst = user_dst;
    if (!dst)
      dst = alloc_tensor(dst_tz);
    assert(tensor_dims(dst) == dst_tz);

    // Create upsampling node and add it to net
    auto node = std::make_shared<Upsample<K>>(src, dst);
    nodes.push_back(node);
    return node;
  }

  template<int K>
  memory::dims Network<K>::concat_dims(const memory::dims& src1_tz, const memory::dims& src2_tz)
  {
    assert(src1_tz[0] == src2_tz[0]); // N
    assert(src1_tz[2] == src2_tz[2]); // H
    assert(src1_tz[3] == src2_tz[3]); // W

    memory::dims dst_tz = src1_tz;
    dst_tz[1] += src2_tz[1]; // C
    return dst_tz;
  }

  template class Network<8>;
  template class Network<16>;

} // ::oidn
