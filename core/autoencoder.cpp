// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "autoencoder.h"
#include "common/tza.h"

// Built-in weights
#include "weights/rt_hdr.h"
#include "weights/rt_hdr_alb.h"
#include "weights/rt_hdr_alb_nrm.h"
#include "weights/rt_ldr.h"
#include "weights/rt_ldr_alb.h"
#include "weights/rt_ldr_alb_nrm.h"
#include "weights/rtlightmap_hdr.h"

namespace oidn {

  // ---------------------------------------------------------------------------
  // AutoencoderFilter
  // ---------------------------------------------------------------------------

  AutoencoderFilter::AutoencoderFilter(const Ref<Device>& device)
    : Filter(device)
  {
  }

  void AutoencoderFilter::setImage(const std::string& name, const Image& data)
  {
    if (name == "color")
      color = data;
    else if (name == "albedo")
      albedo = data;
    else if (name == "normal")
      normal = data;
    else if (name == "output")
      output = data;

    dirty = true;
  }

  void AutoencoderFilter::setData(const std::string& name, const Data& data)
  {
    if (name == "weights")
      userWeights = data;

    dirty = true;
  }

  void AutoencoderFilter::set1i(const std::string& name, int value)
  {
    if (name == "hdr")
      hdr = value;
    else if (name == "srgb")
      srgb = value;
    else if (name == "maxMemoryMB")
      maxMemoryMB = value;

    dirty = true;
  }

  int AutoencoderFilter::get1i(const std::string& name)
  {
    if (name == "hdr")
      return hdr;
    else if (name == "srgb")
      return srgb;
    else if (name == "maxMemoryMB")
      return maxMemoryMB;
    else if (name == "alignment")
      return alignment;
    else if (name == "overlap")
      return overlap;
    else
      throw Exception(Error::InvalidArgument, "invalid parameter");
  }

  void AutoencoderFilter::set1f(const std::string& name, float value)
  {
    if (name == "hdrScale")
      hdrScale = value;

    dirty = true;
  }

  float AutoencoderFilter::get1f(const std::string& name)
  {
    if (name == "hdrScale")
      return hdrScale;
    else
      throw Exception(Error::InvalidArgument, "invalid parameter");
  }

  void AutoencoderFilter::commit()
  {
    if (!dirty)
      return;

    device->executeTask([&]()
    {
      net = buildNet();
    });

    dirty = false;
  }

  void AutoencoderFilter::execute()
  {
    if (dirty)
      throw Exception(Error::InvalidOperation, "changes to the filter are not committed");

    if (!net)
      return;

    device->executeTask([&]()
    {
      Progress progress;
      progress.func = progressFunc;
      progress.userPtr = progressUserPtr;
      progress.taskCount = tileCountH * tileCountW;

      // Iterate over the tiles
      int tileIndex = 0;

      for (int i = 0; i < tileCountH; ++i)
      {
        const int h = i * (tileH - 2*overlap); // input tile position (including overlap)
        const int overlapBeginH = i > 0            ? overlap : 0; // overlap on the top
        const int overlapEndH   = i < tileCountH-1 ? overlap : 0; // overlap on the bottom
        const int tileH1 = min(H - h, tileH); // input tile size (including overlap)
        const int tileH2 = tileH1 - overlapBeginH - overlapEndH; // output tile size
        const int alignOffsetH = tileH - round_up(tileH1, alignment); // align to the bottom in the tile buffer

        for (int j = 0; j < tileCountW; ++j)
        {
          const int w = j * (tileW - 2*overlap); // input tile position (including overlap)
          const int overlapBeginW = j > 0            ? overlap : 0; // overlap on the left
          const int overlapEndW   = j < tileCountW-1 ? overlap : 0; // overlap on the right
          const int tileW1 = min(W - w, tileW); // input tile size (including overlap)
          const int tileW2 = tileW1 - overlapBeginW - overlapEndW; // output tile size
          const int alignOffsetW = tileW - round_up(tileW1, alignment); // align to the right in the tile buffer

          // Set the input tile
          inputReorder->setTile(h, w,
                                alignOffsetH, alignOffsetW,
                                tileH1, tileW1);

          // Set the output tile
          outputReorder->setTile(alignOffsetH + overlapBeginH, alignOffsetW + overlapBeginW,
                                 h + overlapBeginH, w + overlapBeginW,
                                 tileH2, tileW2);

          //printf("Tile: %d %d -> %d %d\n", w+overlapBeginW, h+overlapBeginH, w+overlapBeginW+tileW2, h+overlapBeginH+tileH2);

          // Denoise the tile
          net->execute(progress, tileIndex);

          // Next tile
          tileIndex++;
        }
      }
    });
  }

  void AutoencoderFilter::computeTileSize()
  {
    const int minTileSize = 3*overlap;
    const int estimatedBytesPerPixel = mayiuse(avx512_common) ? estimatedBytesPerPixel16 : estimatedBytesPerPixel8;
    const int64_t maxTilePixels = (int64_t(maxMemoryMB)*1024*1024 - estimatedBytesBase) / estimatedBytesPerPixel;

    tileCountH = 1;
    tileCountW = 1;
    tileH = round_up(H, alignment);
    tileW = round_up(W, alignment);

    // Divide the image into tiles until the tile size gets below the threshold
    while (int64_t(tileH) * tileW > maxTilePixels)
    {
      if (tileH > minTileSize && tileH > tileW)
      {
        tileCountH++;
        tileH = max(round_up(ceil_div(H - 2*overlap, tileCountH), alignment) + 2*overlap, minTileSize);
      }
      else if (tileW > minTileSize)
      {
        tileCountW++;
        tileW = max(round_up(ceil_div(W - 2*overlap, tileCountW), alignment) + 2*overlap, minTileSize);
      }
      else
        break;
    }

    // Compute the final number of tiles
    tileCountH = (H > tileH) ? ceil_div(H - 2*overlap, tileH - 2*overlap) : 1;
    tileCountW = (W > tileW) ? ceil_div(W - 2*overlap, tileW - 2*overlap) : 1;

    if (device->isVerbose(2))
    {
      std::cout << "Tile size : " << tileW << "x" << tileH << std::endl;
      std::cout << "Tile count: " << tileCountW << "x" << tileCountH << std::endl;
    }
  }

  std::shared_ptr<Executable> AutoencoderFilter::buildNet()
  {
    H = color.height;
    W = color.width;

    if (srgb && hdr)
      throw Exception(Error::InvalidOperation, "srgb and hdr modes cannot be enabled at the same time");

    // Get the number of input channels
    int inputC = 0;
    if (color)  inputC += 3;
    if (albedo) inputC += 3;
    if (normal) inputC += 3;

    // Select the weights to use
    Data weights;

    if (userWeights)
      weights = userWeights;
    else if (color && !albedo && !normal)
      weights = hdr ? defaultWeights.hdr : defaultWeights.ldr;
    else if (color && albedo && !normal)
      weights = hdr ? defaultWeights.hdr_alb : defaultWeights.ldr_alb;
    else if (color && albedo && normal)
      weights = hdr ? defaultWeights.hdr_alb_nrm : defaultWeights.ldr_alb_nrm;

    if (!weights)
      throw Exception(Error::InvalidOperation, "unsupported combination of input features");

    // Check the input/output buffers
    if (!output)
      throw Exception(Error::InvalidOperation, "output image not specified");

    if ((color.format != Format::Float3)
        || (albedo && albedo.format != Format::Float3)
        || (normal && normal.format != Format::Float3)
        || (output.format != Format::Float3))
      throw Exception(Error::InvalidOperation, "unsupported image format");

    if ((albedo && (albedo.width != W || albedo.height != H))
        || (normal && (normal.width != W || normal.height != H))
        || (output.width != W || output.height != H))
      throw Exception(Error::InvalidOperation, "image size mismatch");

    if (output.ptr == color.ptr || output.ptr == albedo.ptr || output.ptr == normal.ptr)
      throw Exception(Error::InvalidOperation, "output image is one of the input images");

    // Compute the tile size
    computeTileSize();

    // If the image size is zero, there is nothing else to do
    if (H <= 0 || W <= 0)
      return nullptr;

    // Parse the weights
    const auto weightsMap = parseTZA(weights.ptr, weights.size);

    // Create the network
    std::shared_ptr<Network> net = std::make_shared<Network>(device, weightsMap);

    // Compute the buffer sizes
    const auto inputDims        = memory::dims({1, inputC, tileH, tileW});
    const auto inputReorderDims = net->getInputReorderDims(inputDims, alignment);   //-> concat1

    const auto encConv0Dims  = net->getConvDims("enc_conv0", inputReorderDims);     //-> temp0

    const auto encConv1Dims  = net->getConvDims("enc_conv1", encConv0Dims);         //-> temp1
    const auto pool1Dims     = net->getPoolDims(encConv1Dims);                      //-> concat2

    const auto encConv2Dims  = net->getConvDims("enc_conv2", pool1Dims);            //-> temp0
    const auto pool2Dims     = net->getPoolDims(encConv2Dims);                      //-> concat3

    const auto encConv3Dims  = net->getConvDims("enc_conv3", pool2Dims);            //-> temp0
    const auto pool3Dims     = net->getPoolDims(encConv3Dims);                      //-> concat4

    const auto encConv4Dims  = net->getConvDims("enc_conv4", pool3Dims);            //-> temp0
    const auto pool4Dims     = net->getPoolDims(encConv4Dims);                      //-> concat5

    const auto encConv5Dims  = net->getConvDims("enc_conv5", pool4Dims);            //-> temp0
    const auto pool5Dims     = net->getPoolDims(encConv5Dims);                      //-> temp1

    const auto upsample5Dims = net->getUpsampleDims(pool5Dims);                     //-> concat5
    const auto concat5Dims   = net->getConcatDims(upsample5Dims, pool4Dims);
    const auto decConv5aDims = net->getConvDims("dec_conv5a", concat5Dims);         //-> temp0
    const auto decConv5bDims = net->getConvDims("dec_conv5b", decConv5aDims);       //-> temp1

    const auto upsample4Dims = net->getUpsampleDims(decConv5bDims);                 //-> concat4
    const auto concat4Dims   = net->getConcatDims(upsample4Dims, pool3Dims);
    const auto decConv4aDims = net->getConvDims("dec_conv4a", concat4Dims);         //-> temp0
    const auto decConv4bDims = net->getConvDims("dec_conv4b", decConv4aDims);       //-> temp1

    const auto upsample3Dims = net->getUpsampleDims(decConv4bDims);                 //-> concat3
    const auto concat3Dims   = net->getConcatDims(upsample3Dims, pool2Dims);
    const auto decConv3aDims = net->getConvDims("dec_conv3a", concat3Dims);         //-> temp0
    const auto decConv3bDims = net->getConvDims("dec_conv3b", decConv3aDims);       //-> temp1

    const auto upsample2Dims = net->getUpsampleDims(decConv3bDims);                 //-> concat2
    const auto concat2Dims   = net->getConcatDims(upsample2Dims, pool1Dims);
    const auto decConv2aDims = net->getConvDims("dec_conv2a", concat2Dims);         //-> temp0
    const auto decConv2bDims = net->getConvDims("dec_conv2b", decConv2aDims);       //-> temp1

    const auto upsample1Dims = net->getUpsampleDims(decConv2bDims);                 //-> concat1
    const auto concat1Dims   = net->getConcatDims(upsample1Dims, inputReorderDims);
    const auto decConv1aDims = net->getConvDims("dec_conv1a", concat1Dims);         //-> temp0
    const auto decConv1bDims = net->getConvDims("dec_conv1b", decConv1aDims);       //-> temp1

    const auto decConv0Dims  = net->getConvDims("dec_conv0", decConv1bDims);        //-> temp0

    const auto outputDims = memory::dims({1, 3, tileH, tileW});

    // Allocate two temporary ping-pong buffers to decrease memory usage
    const auto temp0Dims = getMaxMemoryDims({
      encConv0Dims,
      encConv2Dims,
      encConv3Dims,
      encConv4Dims,
      encConv5Dims,
      decConv5aDims,
      decConv4aDims,
      decConv3aDims,
      decConv2aDims,
      decConv1aDims,
      decConv0Dims
    });

    const auto temp1Dims = getMaxMemoryDims({
      encConv1Dims,
      pool5Dims,
      decConv5bDims,
      decConv4bDims,
      decConv3bDims,
      decConv2bDims,
      decConv1bDims,
    });

    auto temp0 = net->allocMemory(temp0Dims);
    auto temp1 = net->allocMemory(temp1Dims);

    // Allocate enough memory to hold the concat outputs. Then use the first
    // half to hold the previous conv output and the second half to hold the
    // pool/orig image output. This works because everything is C dimension
    // outermost, padded to K floats, and all the concats are on the C dimension.
    auto concat1Dst = net->allocMemory(concat1Dims);
    auto concat2Dst = net->allocMemory(concat2Dims);
    auto concat3Dst = net->allocMemory(concat3Dims);
    auto concat4Dst = net->allocMemory(concat4Dims);
    auto concat5Dst = net->allocMemory(concat5Dims);

    // Transfer function
    std::shared_ptr<TransferFunction> transferFunc = makeTransferFunc();

    // Autoexposure
    if (hdr)
    {
      if (isnan(hdrScale))
        net->addAutoexposure(color, transferFunc);
      else
        transferFunc->setExposure(hdrScale);
    }

    // Input reorder
    auto inputReorderDst = net->castMemory(inputReorderDims, concat1Dst, upsample1Dims);
    inputReorder = net->addInputReorder(color, albedo, normal,
                                        transferFunc, hdr,
                                        alignment, inputReorderDst);

    // enc_conv0
    auto encConv0 = net->addConv("enc_conv0", inputReorder->getDst(), temp0);

    // enc_conv1
    auto encConv1 = net->addConv("enc_conv1", encConv0->getDst(), temp1);

    // pool1
    // Adjust pointer for pool1 to eliminate concat1
    auto pool1Dst = net->castMemory(pool1Dims, concat2Dst, upsample2Dims);
    auto pool1 = net->addPool(encConv1->getDst(), pool1Dst);

    // enc_conv2
    auto encConv2 = net->addConv("enc_conv2", pool1->getDst(), temp0);

    // pool2
    // Adjust pointer for pool2 to eliminate concat2
    auto pool2Dst = net->castMemory(pool2Dims, concat3Dst, upsample3Dims);
    auto pool2 = net->addPool(encConv2->getDst(), pool2Dst);

    // enc_conv3
    auto encConv3 = net->addConv("enc_conv3", pool2->getDst(), temp0);

    // pool3
    // Adjust pointer for pool3 to eliminate concat3
    auto pool3Dst = net->castMemory(pool3Dims, concat4Dst, upsample4Dims);
    auto pool3 = net->addPool(encConv3->getDst(), pool3Dst);

    // enc_conv4
    auto encConv4 = net->addConv("enc_conv4", pool3->getDst(), temp0);

    // pool4
    // Adjust pointer for pool4 to eliminate concat4
    auto pool4Dst = net->castMemory(pool4Dims, concat5Dst, upsample5Dims);
    auto pool4 = net->addPool(encConv4->getDst(), pool4Dst);

    // enc_conv5
    auto encConv5 = net->addConv("enc_conv5", pool4->getDst(), temp0);

    // pool5
    auto pool5 = net->addPool(encConv5->getDst(), temp1);

    // upsample5
    auto upsample5Dst = net->castMemory(upsample5Dims, concat5Dst);
    auto upsample5 = net->addUpsample(pool5->getDst(), upsample5Dst);

    // dec_conv5a
    auto decConv5a = net->addConv("dec_conv5a", concat5Dst, temp0);

    // dec_conv5b
    auto decConv5b = net->addConv("dec_conv5b", decConv5a->getDst(), temp1);

    // upsample4
    auto upsample4Dst = net->castMemory(upsample4Dims, concat4Dst);
    auto upsample4 = net->addUpsample(decConv5b->getDst(), upsample4Dst);

    // dec_conv4a
    auto decConv4a = net->addConv("dec_conv4a", concat4Dst, temp0);

    // dec_conv4b
    auto decConv4b = net->addConv("dec_conv4b", decConv4a->getDst(), temp1);

    // upsample3
    auto upsample3Dst = net->castMemory(upsample3Dims, concat3Dst);
    auto upsample3 = net->addUpsample(decConv4b->getDst(), upsample3Dst);

    // dec_conv3a
    auto decConv3a = net->addConv("dec_conv3a", concat3Dst, temp0);

    // dec_conv3b
    auto decConv3b = net->addConv("dec_conv3b", decConv3a->getDst(), temp1);

    // upsample2
    auto upsample2Dst = net->castMemory(upsample2Dims, concat2Dst);
    auto upsample2 = net->addUpsample(decConv3b->getDst(), upsample2Dst);

    // dec_conv2a
    auto decConv2a = net->addConv("dec_conv2a", concat2Dst, temp0);

    // dec_conv2b
    auto decConv2b = net->addConv("dec_conv2b", decConv2a->getDst(), temp1);

    // upsample1
    auto upsample1Dst = net->castMemory(upsample1Dims, concat1Dst);
    auto upsample1 = net->addUpsample(decConv2b->getDst(), upsample1Dst);

    // dec_conv1a
    auto decConv1a = net->addConv("dec_conv1a", concat1Dst, temp0);

    // dec_conv1b
    auto decConv1b = net->addConv("dec_conv1b", decConv1a->getDst(), temp1);

    // dec_conv0
    auto decConv0 = net->addConv("dec_conv0", decConv1b->getDst(), temp0, false /* no relu */);

    // Output reorder
    outputReorder = net->addOutputReorder(decConv0->getDst(), transferFunc, hdr, output);

    net->finalize();
    return net;
  }

  std::shared_ptr<TransferFunction> AutoencoderFilter::makeTransferFunc()
  {
    if (hdr)
      return std::make_shared<TransferFunction>(TransferFunction::Type::PU);
    else if (srgb)
      return std::make_shared<TransferFunction>(TransferFunction::Type::Linear);
    else
      return std::make_shared<TransferFunction>(TransferFunction::Type::SRGB);
  }

  // ---------------------------------------------------------------------------
  // RTFilter
  // ---------------------------------------------------------------------------

  RTFilter::RTFilter(const Ref<Device>& device)
    : AutoencoderFilter(device)
  {
    defaultWeights.ldr         = blobs::weights::rt_ldr;
    defaultWeights.ldr_alb     = blobs::weights::rt_ldr_alb;
    defaultWeights.ldr_alb_nrm = blobs::weights::rt_ldr_alb_nrm;
    defaultWeights.hdr         = blobs::weights::rt_hdr;
    defaultWeights.hdr_alb     = blobs::weights::rt_hdr_alb;
    defaultWeights.hdr_alb_nrm = blobs::weights::rt_hdr_alb_nrm;
  }

  // ---------------------------------------------------------------------------
  // RTLightmapFilter
  // ---------------------------------------------------------------------------

  RTLightmapFilter::RTLightmapFilter(const Ref<Device>& device)
    : AutoencoderFilter(device)
  {
    defaultWeights.hdr = blobs::weights::rtlightmap_hdr;

    hdr = true;
  }

  std::shared_ptr<TransferFunction> RTLightmapFilter::makeTransferFunc()
  {
    return std::make_shared<TransferFunction>(TransferFunction::Type::Log);
  }

} // namespace oidn
