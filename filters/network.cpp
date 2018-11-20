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
    return cast_tensor(dims, src, get_tensor_size(src_offset));
  }

  template<int K>
  void Network<K>::zero_tensor(const std::shared_ptr<memory>& dst)
  {
    assert(tensor_type(dst) == memory::data_type::f32);
    memset(dst->get_data_handle(), 0, get_tensor_size(dst)*sizeof(float));
  }

  template<int K>
  void Network<K>::weights_reorder(const std::shared_ptr<memory>& src,
                                   const std::shared_ptr<memory>& dst)
  {
    WeightsReorder<K>(src, dst).execute();
  }

  template<int K>
  memory::dims Network<K>::get_input_reorder_dims(const memory::dims& src_dims, int spatial_pad)
  {
    memory::dims dst_dims = src_dims;
    dst_dims[1] = get_padded<K>(src_dims[1]); // round up C
    dst_dims[2] = (src_dims[2] + spatial_pad - 1) / spatial_pad * spatial_pad; // round up H
    dst_dims[3] = (src_dims[3] + spatial_pad - 1) / spatial_pad * spatial_pad; // round up W
    return dst_dims;
  }

  template<int K>
  memory::dims Network<K>::get_conv_dims(const std::string& name, const memory::dims& src_dims)
  {
    auto b = weight_map[name + "/b"];
    memory::dims dst_dims = src_dims;
    dst_dims[1] = get_padded<K>(b.dims[0]); // dst_dims[C] = get_padded(OC)
    return dst_dims;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::add_conv(const std::string& name,
                                             const std::shared_ptr<memory>& src,
                                             bool relu)
  {
    const memory::dims strides = {1, 1};
    const memory::dims padding = {1, 1};

    memory::dims src_dims = get_tensor_dims(src);

    // Get the weights
    const auto& W = weight_map[name + "/W"];
    if (W.ndims() != 4 || W.format != "oihw")
      throw std::runtime_error("invalid weights");
    memory::dims weights_dims = W.dims;
    auto user_weights = alloc_tensor(weights_dims, memory::format::oihw, W.data);

    // Reorder/pad the weights
    memory::dims weights_pad_dims = weights_dims;
    weights_pad_dims[1] = get_padded<K>(weights_dims[1]); // IC
    weights_pad_dims[0] = get_padded<K>(weights_dims[0]); // OC
    assert(src_dims[1] == weights_pad_dims[1]); // src_dims[C] == weights_pad_dims[IC]
    auto weights = alloc_tensor(weights_pad_dims, BlockedFormat<K>::OIhwKiKo);
    weights_reorder(user_weights, weights);

    // Get the biases
    const auto& b = weight_map[name + "/b"];
    if (b.ndims() != 1)
      throw std::runtime_error("invalid biases");
    memory::dims bias_dims = b.dims;

    // Copy/pad the biases
    memory::dims bias_pad_dims = {get_padded<K>(bias_dims[0])};
    auto bias = alloc_tensor(bias_pad_dims);
    if (bias_dims[0] != bias_pad_dims[0])
      memset(bias->get_data_handle(), 0, bias_pad_dims[0]*sizeof(float));
    memcpy(bias->get_data_handle(), b.data, bias_dims[0]*sizeof(float));

    // Allocate memory for destination
    memory::dims dst_dims = src_dims;
    dst_dims[1] = weights_pad_dims[0]; // dst_dims[C] = weights_pad_dims[OC]
    auto dst = alloc_tensor(dst_dims);
    assert(get_tensor_dims(dst) == dst_dims);

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
  memory::dims Network<K>::get_pool_dims(const memory::dims& src_dims)
  {
    memory::dims dst_dims = src_dims;
    dst_dims[2] /= 2; // H/2
    dst_dims[3] /= 2; // W/2
    return dst_dims;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::add_pool(const std::shared_ptr<memory>& src,
                                             const std::shared_ptr<memory>& user_dst)
  {
    const memory::dims kernel  = {2, 2};
    const memory::dims strides = {2, 2};
    const memory::dims padding = {0, 0};

    memory::dims src_dims = get_tensor_dims(src);
    memory::dims dst_dims = get_pool_dims(src_dims);

    auto dst = user_dst;
    if (!dst)
      dst = alloc_tensor(dst_dims);
    assert(get_tensor_dims(dst) == dst_dims);

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
  memory::dims Network<K>::get_unpool_dims(const memory::dims& src_dims)
  {
    memory::dims dst_dims = src_dims;
    dst_dims[2] *= 2; // H*2
    dst_dims[3] *= 2; // W*2
    return dst_dims;
  }

  template<int K>
  std::shared_ptr<Node> Network<K>::add_unpool(const std::shared_ptr<memory>& src,
                                               const std::shared_ptr<memory>& user_dst)
  {
    memory::dims src_dims = get_tensor_dims(src);
    memory::dims dst_dims = get_unpool_dims(src_dims);

    auto dst = user_dst;
    if (!dst)
      dst = alloc_tensor(dst_dims);
    assert(get_tensor_dims(dst) == dst_dims);

    // Create upsampling node and add it to net
    auto node = std::make_shared<Upsample<K>>(src, dst);
    nodes.push_back(node);
    return node;
  }

  template<int K>
  memory::dims Network<K>::get_concat_dims(const memory::dims& src1_dims, const memory::dims& src2_dims)
  {
    assert(src1_dims[0] == src2_dims[0]); // N
    assert(src1_dims[2] == src2_dims[2]); // H
    assert(src1_dims[3] == src2_dims[3]); // W

    memory::dims dst_dims = src1_dims;
    dst_dims[1] += src2_dims[1]; // C
    return dst_dims;
  }

  template class Network<8>;
  template class Network<16>;

} // ::oidn
