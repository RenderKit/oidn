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

#include "autoencoder.h"

namespace oidn {

  // -------------------------------------------------------------------------
  // AutoencoderFilter
  // -------------------------------------------------------------------------

  AutoencoderFilter::AutoencoderFilter(const Ref<Device>& device)
    : Filter(device),
      srgb(false),
      hdr(false),
      weightData { nullptr, nullptr }
  {
  }

  void AutoencoderFilter::setImage(const std::string& name, const Image& data)
  {
    if (name == "color")
    {
      if (data.format != Format::Float3)
        throw Exception(Error::InvalidOperation, "unsupported image format");
      color = data;
    }
    else if (name == "albedo")
    {
      if (data.format != Format::Float3)
        throw Exception(Error::InvalidOperation, "unsupported image format");
      albedo = data;
    }
    else if (name == "normal")
    {
      if (data.format != Format::Float3)
        throw Exception(Error::InvalidOperation, "unsupported image format");
      normal = data;
    }
    else if (name == "output")
    {
      if (data.format != Format::Float3)
        throw Exception(Error::InvalidOperation, "unsupported image format");
      output = data;
    }
  }

  void AutoencoderFilter::set1i(const std::string& name, int value)
  {
    if (name == "srgb")
      srgb = value;
    else if (name == "hdr")
      hdr = value;
  }

  void AutoencoderFilter::commit()
  {
    device->executeTask([&]()
    {
      if (mayiuse(avx512_common))
        net = buildNet<16>();
      else
        net = buildNet<8>();
    });
  }

  void AutoencoderFilter::execute()
  {
    if (!isCommitted())
      throw Exception(Error::InvalidOperation, "the filter is not committed");

    device->executeTask([&]()
    {
      if (hdr)
      {
        const float exposure = autoexposure(color);
        //printf("exposure = %f\n", exposure);
        std::static_pointer_cast<HDRTransferFunction>(transferFunc)->setExposure(exposure);
      }

      net->execute();
    });
  }

  template<int K>
  std::shared_ptr<Node> AutoencoderFilter::buildNet()
  {
    constexpr int spatialPad = 32; // the image must be padded spatially

    const int width = color.width;
    const int height = color.height;

    // Configure the network
    int inputC;
    void* weightPtr;

    if (srgb && hdr)
      throw Exception(Error::InvalidOperation, "srgb and hdr modes cannot be enabled at the same time");

    if (color && !albedo && !normal && weightData.hdr)
    {
      inputC = 3;
      weightPtr = hdr ? weightData.hdr : weightData.ldr;
    }
    else if (color && albedo && !normal && weightData.hdr_alb)
    {
      inputC = 6;
      weightPtr = hdr ? weightData.hdr_alb : weightData.ldr_alb;
    }
    else if (color && albedo && normal && weightData.hdr_alb_nrm)
    {
      inputC = 9;
      weightPtr = hdr ? weightData.hdr_alb_nrm : weightData.ldr_alb_nrm;
    }
    else
    {
      throw Exception(Error::InvalidOperation, "unsupported combination of input features");
    }

    if (!output)
      throw Exception(Error::InvalidOperation, "output image not specified");
    if ((albedo && (albedo.width != width || albedo.height != height)) ||
        (normal && (normal.width != width || normal.height != height)) ||
        (output.width != width || output.height != height))
      throw Exception(Error::InvalidOperation, "image size mismatch");

    // Parse the weights
    const auto weightMap = parseTensors(weightPtr);

    // Create the network
    std::shared_ptr<Network<K>> net = std::make_shared<Network<K>>(weightMap);

    // Compute the tensor sizes
    const auto inputDims        = memory::dims({1, inputC, height, width});
    const auto inputReorderDims = net->getInputReorderDims(inputDims, spatialPad);

    const auto conv1Dims   = net->getConvDims("conv1", inputReorderDims);
    const auto conv1bDims  = net->getConvDims("conv1b", conv1Dims);
    const auto pool1Dims   = net->getPoolDims(conv1bDims);
    const auto conv2Dims   = net->getConvDims("conv2", pool1Dims);
    const auto pool2Dims   = net->getPoolDims(conv2Dims);
    const auto conv3Dims   = net->getConvDims("conv3", pool2Dims);
    const auto pool3Dims   = net->getPoolDims(conv3Dims);
    const auto conv4Dims   = net->getConvDims("conv4", pool3Dims);
    const auto pool4Dims   = net->getPoolDims(conv4Dims);
    const auto conv5Dims   = net->getConvDims("conv5", pool4Dims);
    const auto pool5Dims   = net->getPoolDims(conv5Dims);
    const auto unpool4Dims = net->getUnpoolDims(pool5Dims);
    const auto concat4Dims = net->getConcatDims(unpool4Dims, pool4Dims);
    const auto conv6Dims   = net->getConvDims("conv6", concat4Dims);
    const auto conv6bDims  = net->getConvDims("conv6b", conv6Dims);
    const auto unpool3Dims = net->getUnpoolDims(conv6bDims);
    const auto concat3Dims = net->getConcatDims(unpool3Dims, pool3Dims);
    const auto conv7Dims   = net->getConvDims("conv7", concat3Dims);
    const auto conv7bDims  = net->getConvDims("conv7b", conv7Dims);
    const auto unpool2Dims = net->getUnpoolDims(conv7bDims);
    const auto concat2Dims = net->getConcatDims(unpool2Dims, pool2Dims);
    const auto conv8Dims   = net->getConvDims("conv8", concat2Dims);
    const auto conv8bDims  = net->getConvDims("conv8b", conv8Dims);
    const auto unpool1Dims = net->getUnpoolDims(conv8bDims);
    const auto concat1Dims = net->getConcatDims(unpool1Dims, pool1Dims);
    const auto conv9Dims   = net->getConvDims("conv9", concat1Dims);
    const auto conv9bDims  = net->getConvDims("conv9b", conv9Dims);
    const auto unpool0Dims = net->getUnpoolDims(conv9bDims);
    const auto concat0Dims = net->getConcatDims(unpool0Dims, inputReorderDims);
    const auto conv10Dims  = net->getConvDims("conv10", concat0Dims);
    const auto conv10bDims = net->getConvDims("conv10b", conv10Dims);
    const auto conv11Dims  = net->getConvDims("conv11", conv10bDims);

    const auto outputDims = memory::dims({1, 3, height, width});

    // Allocate enough memory to hold the concat outputs. Then use the first
    // half to hold the previous conv output and the second half to hold the
    // pool/orig image output. This works because everything is C dimension
    // outermost, padded to K floats, and all the concats are on the C dimension.
    auto concat0Dst = net->allocTensor(concat0Dims);
    auto concat1Dst = net->allocTensor(concat1Dims);
    auto concat2Dst = net->allocTensor(concat2Dims);
    auto concat3Dst = net->allocTensor(concat3Dims);
    auto concat4Dst = net->allocTensor(concat4Dims);

    // Input reorder
    auto inputReorderDst = net->castTensor(inputReorderDims, concat0Dst, unpool0Dims);
    std::shared_ptr<Node> inputReorder;
    if (srgb)
    {
      transferFunc = std::make_shared<LinearTransferFunction>();
      inputReorder = net->addInputReorder(color, albedo, normal,
                                          std::static_pointer_cast<LinearTransferFunction>(transferFunc),
                                          spatialPad, inputReorderDst);
    }
    else if (hdr)
    {
      transferFunc = std::make_shared<HDRTransferFunction>();
      inputReorder = net->addInputReorder(color, albedo, normal,
                                          std::static_pointer_cast<HDRTransferFunction>(transferFunc),
                                          spatialPad, inputReorderDst);
    }
    else
    {
      transferFunc = std::make_shared<SRGBTransferFunction>();
      inputReorder = net->addInputReorder(color, albedo, normal,
                                          std::static_pointer_cast<SRGBTransferFunction>(transferFunc),
                                          spatialPad, inputReorderDst);
    }

    // conv1
    auto conv1 = net->addConv("conv1", inputReorder->getDst());

    // conv1b
    auto conv1b = net->addConv("conv1b", conv1->getDst());

    // pool1
    // Adjust pointer for pool1 to eliminate concat1
    auto pool1Dst = net->castTensor(pool1Dims, concat1Dst, unpool1Dims);
    auto pool1 = net->addPool(conv1b->getDst(), pool1Dst);

    // conv2
    auto conv2 = net->addConv("conv2", pool1->getDst());

    // pool2
    // Adjust pointer for pool2 to eliminate concat2
    auto pool2Dst = net->castTensor(pool2Dims, concat2Dst, unpool2Dims);
    auto pool2 = net->addPool(conv2->getDst(), pool2Dst);

    // conv3
    auto conv3 = net->addConv("conv3", pool2->getDst());

    // pool3
    // Adjust pointer for pool3 to eliminate concat3
    auto pool3Dst = net->castTensor(pool3Dims, concat3Dst, unpool3Dims);
    auto pool3 = net->addPool(conv3->getDst(), pool3Dst);

    // conv4
    auto conv4 = net->addConv("conv4", pool3->getDst());

    // pool4
    // Adjust pointer for pool4 to eliminate concat4
    auto pool4Dst = net->castTensor(pool4Dims, concat4Dst, unpool4Dims);
    auto pool4 = net->addPool(conv4->getDst(), pool4Dst);

    // conv5
    auto conv5 = net->addConv("conv5", pool4->getDst());

    // pool5
    auto pool5 = net->addPool(conv5->getDst());

    // unpool4
    auto unpool4Dst = net->castTensor(unpool4Dims, concat4Dst);
    auto unpool4 = net->addUnpool(pool5->getDst(), unpool4Dst);

    // conv6
    auto conv6 = net->addConv("conv6", concat4Dst);

    // conv6b
    auto conv6b = net->addConv("conv6b", conv6->getDst());

    // unpool3
    auto unpool3Dst = net->castTensor(unpool3Dims, concat3Dst);
    auto unpool3 = net->addUnpool(conv6b->getDst(), unpool3Dst);

    // conv7
    auto conv7 = net->addConv("conv7", concat3Dst);

    // conv7b
    auto conv7b = net->addConv("conv7b", conv7->getDst());

    // unpool2
    auto unpool2Dst = net->castTensor(unpool2Dims, concat2Dst);
    auto unpool2 = net->addUnpool(conv7b->getDst(), unpool2Dst);

    // conv8
    auto conv8 = net->addConv("conv8", concat2Dst);

    // conv8b
    auto conv8b = net->addConv("conv8b", conv8->getDst());

    // unpool1
    auto unpool1Dst = net->castTensor(unpool1Dims, concat1Dst);
    auto unpool1 = net->addUnpool(conv8b->getDst(), unpool1Dst);

    // conv9
    auto conv9 = net->addConv("conv9", concat1Dst);

    // conv9b
    auto conv9b = net->addConv("conv9b", conv9->getDst());

    // unpool0
    auto unpool0Dst = net->castTensor(unpool0Dims, concat0Dst);
    auto unpool0 = net->addUnpool(conv9b->getDst(), unpool0Dst);

    // conv10
    auto conv10 = net->addConv("conv10", concat0Dst);

    // conv10b
    auto conv10b = net->addConv("conv10b", conv10->getDst());

    // conv11
    auto conv11 = net->addConv("conv11", conv10b->getDst(), false /* no relu */);

    // Output reorder
    if (srgb)
      net->addOutputReorder(conv11->getDst(), std::static_pointer_cast<LinearTransferFunction>(transferFunc), output);
    else if (hdr)
      net->addOutputReorder(conv11->getDst(), std::static_pointer_cast<HDRTransferFunction>(transferFunc), output);
    else
      net->addOutputReorder(conv11->getDst(), std::static_pointer_cast<SRGBTransferFunction>(transferFunc), output);

    return net;
  }

  // -------------------------------------------------------------------------
  // RTFilter
  // -------------------------------------------------------------------------

  namespace weights
  {
    // LDR
    extern unsigned char rt_ldr[];         // color
    extern unsigned char rt_ldr_alb[];     // color, albedo
    extern unsigned char rt_ldr_alb_nrm[]; // color, albedo, normal

    // HDR
    extern unsigned char rt_hdr[];         // color
    extern unsigned char rt_hdr_alb[];     // color, albedo
    extern unsigned char rt_hdr_alb_nrm[]; // color, albedo, normal
  }

  RTFilter::RTFilter(const Ref<Device>& device)
    : AutoencoderFilter(device)
  {
    weightData.ldr         = weights::rt_ldr;
    weightData.ldr_alb     = weights::rt_ldr_alb;
    weightData.ldr_alb_nrm = weights::rt_ldr_alb_nrm;
    weightData.hdr         = weights::rt_hdr;
    weightData.hdr_alb     = weights::rt_hdr_alb;
    weightData.hdr_alb_nrm = weights::rt_hdr_alb_nrm;
  }

} // namespace oidn
