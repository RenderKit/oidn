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

#include "autoencoder.h"

namespace oidn {

  // Trained weights stored in binary blobs
  namespace Weights
  {
    extern unsigned char ae_ldr_albedo_normal[];
    extern const size_t ae_ldr_albedo_normal_size;
  }

  void Autoencoder::set_buffer(BufferType type, int slot, const BufferView2D& view)
  {
    if (slot != 0)
      throw std::invalid_argument("invalid buffer slot");

    switch (type)
    {
    case BufferType::INPUT:
      if (view.format != Format::FLOAT3 && view.format != Format::FLOAT3_SRGB)
        throw std::invalid_argument("invalid buffer format");
      input = view;
      break;

    case BufferType::INPUT_ALBEDO:
      if (view.format != Format::FLOAT3)
        throw std::invalid_argument("invalid buffer format");
      input_albedo = view;
      break;

    case BufferType::INPUT_NORMAL:
      if (view.format != Format::FLOAT3)
        throw std::invalid_argument("invalid buffer format");
      input_normal = view;
      break;

    case BufferType::OUTPUT:
      if (view.format != Format::FLOAT3 && view.format != Format::FLOAT3_SRGB)
        throw std::invalid_argument("invalid buffer format");
      output = view;
      break;
    }
  }

  void Autoencoder::commit()
  {
    if (!input || !input_albedo || !input_normal)
      throw std::runtime_error("input buffer(s) not specified");
    if (!output)
      throw std::runtime_error("output buffer not specified");
    if (input_albedo.width != input.width || input_albedo.height != input.height ||
        input_normal.width != input.width || input_normal.height != input.height ||
        output.width != input.width || output.height != input.height)
      throw std::runtime_error("buffer size mismatch");

    device->execute_task([&]()
    {
      if (mayiuse(avx512_common))
        net = build_net<16>();
      else
        net = build_net<8>();
    });
  }

  void Autoencoder::execute()
  {
    device->execute_task([&]() { net->execute(); });
  }

  template<int K>
  std::shared_ptr<Node> Autoencoder::build_net()
  {
    constexpr int spatial_pad = 32; // the image must be padded spatially

    int width = input.width;
    int height = input.height;

    // Parse the weights
    void* weights = Weights::ae_ldr_albedo_normal;
    auto weight_map = parse_tensors(weights);

    // Create the network
    std::shared_ptr<Network<K>> net = std::make_shared<Network<K>>(weight_map);

    // Compute the tensor sizes
    const auto input_tz       = memory::dims({1, 9, height, width});
    const auto input_pad_tz   = net->input_reorder_dims(input_tz, spatial_pad);
    const auto conv1_dst_tz   = net->conv_dims("conv1", input_pad_tz);
    const auto conv1b_dst_tz  = net->conv_dims("conv1b", conv1_dst_tz);
    const auto pool1_dst_tz   = net->pool_dims(conv1b_dst_tz);
    const auto conv2_dst_tz   = net->conv_dims("conv2", pool1_dst_tz);
    const auto pool2_dst_tz   = net->pool_dims(conv2_dst_tz);
    const auto conv3_dst_tz   = net->conv_dims("conv3", pool2_dst_tz);
    const auto pool3_dst_tz   = net->pool_dims(conv3_dst_tz);
    const auto conv4_dst_tz   = net->conv_dims("conv4", pool3_dst_tz);
    const auto pool4_dst_tz   = net->pool_dims(conv4_dst_tz);
    const auto conv5_dst_tz   = net->conv_dims("conv5", pool4_dst_tz);
    const auto pool5_dst_tz   = net->pool_dims(conv5_dst_tz);
    const auto unpool4_dst_tz = net->unpool_dims(pool5_dst_tz);
    const auto concat4_dst_tz = net->concat_dims(unpool4_dst_tz, pool4_dst_tz);
    const auto conv6_dst_tz   = net->conv_dims("conv6", concat4_dst_tz);
    const auto conv6b_dst_tz  = net->conv_dims("conv6b", conv6_dst_tz);
    const auto unpool3_dst_tz = net->unpool_dims(conv6b_dst_tz);
    const auto concat3_dst_tz = net->concat_dims(unpool3_dst_tz, pool3_dst_tz);
    const auto conv7_dst_tz   = net->conv_dims("conv7", concat3_dst_tz);
    const auto conv7b_dst_tz  = net->conv_dims("conv7b", conv7_dst_tz);
    const auto unpool2_dst_tz = net->unpool_dims(conv7b_dst_tz);
    const auto concat2_dst_tz = net->concat_dims(unpool2_dst_tz, pool2_dst_tz);
    const auto conv8_dst_tz   = net->conv_dims("conv8", concat2_dst_tz);
    const auto conv8b_dst_tz  = net->conv_dims("conv8b", conv8_dst_tz);
    const auto unpool1_dst_tz = net->unpool_dims(conv8b_dst_tz);
    const auto concat1_dst_tz = net->concat_dims(unpool1_dst_tz, pool1_dst_tz);
    const auto conv9_dst_tz   = net->conv_dims("conv9", concat1_dst_tz);
    const auto conv9b_dst_tz  = net->conv_dims("conv9b", conv9_dst_tz);
    const auto unpool0_dst_tz = net->unpool_dims(conv9b_dst_tz);
    const auto concat0_dst_tz = net->concat_dims(unpool0_dst_tz, input_pad_tz);
    const auto conv10_dst_tz  = net->conv_dims("conv10", concat0_dst_tz);
    const auto conv10b_dst_tz = net->conv_dims("conv10b", conv10_dst_tz);
    const auto conv11_dst_tz  = net->conv_dims("conv11", conv10b_dst_tz);
    const auto output_tz      = memory::dims({1, 3, height, width});

    // Allocate enough memory to hold the concat outputs. Then use the first
    // half to hold the previous conv output and the second half to hold the
    // pool/orig image output. This works because everything is C dimension
    // outermost, padded to K floats, and all the concats are on the C dimension.
    auto concat0_dst = net->alloc_tensor(concat0_dst_tz);
    auto concat1_dst = net->alloc_tensor(concat1_dst_tz);
    auto concat2_dst = net->alloc_tensor(concat2_dst_tz);
    auto concat3_dst = net->alloc_tensor(concat3_dst_tz);
    auto concat4_dst = net->alloc_tensor(concat4_dst_tz);

    // Input reorder
    auto input_reorder_dst = net->cast_tensor(input_pad_tz, concat0_dst, unpool0_dst_tz);
    auto input_reorder = net->add_input_reorder(input, input_albedo, input_normal, spatial_pad, input_reorder_dst);

    // conv1
    auto conv1 = net->add_conv("conv1", input_reorder->get_dst());

    // conv1b
    auto conv1b = net->add_conv("conv1b", conv1->get_dst());

    // pool1
    // Adjust pointer for pool1 to eliminate concat1
    auto pool1_dst = net->cast_tensor(pool1_dst_tz, concat1_dst, unpool1_dst_tz);
    auto pool1 = net->add_pool(conv1b->get_dst(), pool1_dst);

    // conv2
    auto conv2 = net->add_conv("conv2", pool1->get_dst());

    // pool2
    // Adjust pointer for pool2 to eliminate concat2
    auto pool2_dst = net->cast_tensor(pool2_dst_tz, concat2_dst, unpool2_dst_tz);
    auto pool2 = net->add_pool(conv2->get_dst(), pool2_dst);

    // conv3
    auto conv3 = net->add_conv("conv3", pool2->get_dst());

    // pool3
    // Adjust pointer for pool3 to eliminate concat3
    auto pool3_dst = net->cast_tensor(pool3_dst_tz, concat3_dst, unpool3_dst_tz);
    auto pool3 = net->add_pool(conv3->get_dst(), pool3_dst);

    // conv4
    auto conv4 = net->add_conv("conv4", pool3->get_dst());

    // pool4
    // Adjust pointer for pool4 to eliminate concat4
    auto pool4_dst = net->cast_tensor(pool4_dst_tz, concat4_dst, unpool4_dst_tz);
    auto pool4 = net->add_pool(conv4->get_dst(), pool4_dst);

    // conv5
    auto conv5 = net->add_conv("conv5", pool4->get_dst());

    // pool5
    auto pool5 = net->add_pool(conv5->get_dst());

    // unpool4
    auto unpool4_dst = net->cast_tensor(unpool4_dst_tz, concat4_dst);
    auto unpool4 = net->add_unpool(pool5->get_dst(), unpool4_dst);

    // conv6
    auto conv6 = net->add_conv("conv6", concat4_dst);

    // conv6b
    auto conv6b = net->add_conv("conv6b", conv6->get_dst());

    // unpool3
    auto unpool3_dst = net->cast_tensor(unpool3_dst_tz, concat3_dst);
    auto unpool3 = net->add_unpool(conv6b->get_dst(), unpool3_dst);

    // conv7
    auto conv7 = net->add_conv("conv7", concat3_dst);

    // conv7b
    auto conv7b = net->add_conv("conv7b", conv7->get_dst());

    // unpool2
    auto unpool2_dst = net->cast_tensor(unpool2_dst_tz, concat2_dst);
    auto unpool2 = net->add_unpool(conv7b->get_dst(), unpool2_dst);

    // conv8
    auto conv8 = net->add_conv("conv8", concat2_dst);

    // conv8b
    auto conv8b = net->add_conv("conv8b", conv8->get_dst());

    // unpool1
    auto unpool1_dst = net->cast_tensor(unpool1_dst_tz, concat1_dst);
    auto unpool1 = net->add_unpool(conv8b->get_dst(), unpool1_dst);

    // conv9
    auto conv9 = net->add_conv("conv9", concat1_dst);

    // conv9b
    auto conv9b = net->add_conv("conv9b", conv9->get_dst());

    // unpool0
    auto unpool0_dst = net->cast_tensor(unpool0_dst_tz, concat0_dst);
    auto unpool0 = net->add_unpool(conv9b->get_dst(), unpool0_dst);

    // conv10
    auto conv10 = net->add_conv("conv10", concat0_dst);

    // conv10b
    auto conv10b = net->add_conv("conv10b", conv10->get_dst());

    // conv11
    auto conv11 = net->add_conv("conv11", conv10b->get_dst(), false /* no relu */);

    // Output reorder
    net->add_output_reorder(conv11->get_dst(), output);

    return net;
  }

} // ::oidn
