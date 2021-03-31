// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tza.h"
#include "output_copy.h"
#include "unet.h"

// Built-in weights
#include "weights/rt_hdr.h"
#include "weights/rt_hdr_alb.h"
#include "weights/rt_hdr_alb_nrm.h"
#include "weights/rt_hdr_calb_cnrm.h"
#include "weights/rt_ldr.h"
#include "weights/rt_ldr_alb.h"
#include "weights/rt_ldr_alb_nrm.h"
#include "weights/rtlightmap_hdr.h"
#include "weights/rtlightmap_dir.h"

namespace oidn {

  // ---------------------------------------------------------------------------
  // UNetFilter
  // ---------------------------------------------------------------------------

  UNetFilter::UNetFilter(const Ref<Device>& device)
    : Filter(device)
  {
  }

  void UNetFilter::setData(const std::string& name, const Data& data)
  {
    if (name == "weights")
      userWeights = data;
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void UNetFilter::set1f(const std::string& name, float value)
  {
    if (name == "inputScale" || name == "hdrScale")
      inputScale = value;
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  float UNetFilter::get1f(const std::string& name)
  {
    if (name == "inputScale" || name == "hdrScale")
      return inputScale;
    else
      throw Exception(Error::InvalidArgument, "unknown filter parameter");
  }

  void UNetFilter::commit()
  {
    if (!dirty)
      return;

    device->executeTask([&]()
    {
      net = buildNet();
    });

    dirty = false;
  }

  void UNetFilter::execute()
  {
    if (dirty)
      throw Exception(Error::InvalidOperation, "changes to the filter are not committed");

    if (!net)
      return;

    device->executeTask([&]()
    {
      // Initialize the progress state
      double workAmount = tileCountH * tileCountW * net->getWorkAmount();
      if (outputTemp)
        workAmount += 1;
      Progress progress(progressFunc, progressUserPtr, workAmount);

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
          net->execute(progress);

          // Next tile
          tileIndex++;
        }
      }

      // Copy the output image to the final buffer if filtering in-place
      if (outputTemp)
        outputCopy(outputTemp, output);

      // Finished
      progress.finish();
    });
  }

  void UNetFilter::computeTileSize()
  {
    const int minTileSize = 3*overlap;

    // Estimate the required amount of memory
    int totalEstimatedBytesPerPixel = estimatedBytesPerPixel;
    if (inplace)
      totalEstimatedBytesPerPixel += getByteSize(output.format); // outputTemp

    // Determine the maximum allowed tile size to fit into the requested memory limit
    const int64_t maxTilePixels = (int64_t(maxMemoryMB)*1024*1024 - estimatedBytesBase) / totalEstimatedBytesPerPixel;

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
      std::cout << "Image size: " << W << "x" << H << std::endl;
      std::cout << "Tile size : " << tileW << "x" << tileH << std::endl;
      std::cout << "Tile count: " << tileCountW << "x" << tileCountH << std::endl;
      std::cout << "In-place  : " << (inplace ? "true" : "false") << std::endl;
    }
  }

  Ref<Network> UNetFilter::buildNet()
  {
    // Check the input/output buffers
    if (!color && !albedo && !normal)
      throw Exception(Error::InvalidOperation, "input image not specified");
    if (!output)
      throw Exception(Error::InvalidOperation, "output image not specified");

    H = output.height;
    W = output.width;

    if ((color  && color.format  != Format::Float3) ||
        (albedo && albedo.format != Format::Float3) ||
        (normal && normal.format != Format::Float3) ||
        (output.format != Format::Float3))
      throw Exception(Error::InvalidOperation, "unsupported image format");

    if ((color  && (color.width  != W || color.height  != H)) ||
        (albedo && (albedo.width != W || albedo.height != H)) ||
        (normal && (normal.width != W || normal.height != H)))
      throw Exception(Error::InvalidOperation, "image size mismatch");

    if (srgb && hdr)
      throw Exception(Error::InvalidOperation, "srgb and hdr modes cannot be enabled at the same time");

    // Get the number of input channels
    int inputC = 0;
    if (color)  inputC += 3;
    if (albedo) inputC += 3;
    if (normal) inputC += 3;

    if (device->isVerbose(2))
    {
      std::cout << "Inputs:";
      if (color)  std::cout << " " << (directional ? "dir" : (hdr ? "hdr" : "ldr"));
      if (albedo) std::cout << " " << "alb";
      if (normal) std::cout << " " << "nrm";
      std::cout << std::endl;
    }

    // Select the weights to use
    Data weights;

    if (userWeights)
      weights = userWeights;
    else if (color && !albedo && !normal)
      weights = directional ? builtinWeights.dir : (hdr ? builtinWeights.hdr : builtinWeights.ldr);
    else if (color && albedo && !normal)
      weights = hdr ? builtinWeights.hdr_alb : builtinWeights.ldr_alb;
    else if (color && albedo && normal)
    {
      if (hdr)
        weights = cleanAux ? builtinWeights.hdr_calb_cnrm : builtinWeights.hdr_alb_nrm;
      else
        weights = builtinWeights.ldr_alb_nrm;
    }

    if (!weights)
      throw Exception(Error::InvalidOperation, "unsupported combination of input features");

    const bool snorm = directional || (!color && !albedo && normal);

    // Determine whether in-place filtering is required
    inplace = output.overlaps(color)  ||
              output.overlaps(albedo) ||
              output.overlaps(normal);

    // Compute the tile size
    computeTileSize();

    // If the image size is zero, there is nothing else to do
    if (H <= 0 || W <= 0)
      return nullptr;

    // If doing in-place _tiled_ filtering, allocate a temporary output buffer
    // For non-tiled filtering this is not necessary as we use ping-pong buffers
    if (inplace && (tileCountH * tileCountW) > 1)
      outputTemp = Image(device, output.format, W, H);

    // Parse the weights blob
    const auto weightsMap = parseTZA(device, weights.ptr, weights.size);

    // Create the network
    Ref<Network> net = makeRef<Network>(device, weightsMap);

    // Compute the buffer sizes
    const auto inputDims        = TensorDims({inputC, tileH, tileW});
    const auto inputReorderDims = net->getInputReorderDims(inputDims, alignment); //-> concat1

    const auto encConv0Dims  = net->getConvDims("enc_conv0", inputReorderDims);   //-> temp0

    const auto encConv1Dims  = net->getConvDims("enc_conv1", encConv0Dims);       //-> temp1
    const auto pool1Dims     = net->getPoolDims(encConv1Dims);                    //-> concat2

    const auto encConv2Dims  = net->getConvDims("enc_conv2", pool1Dims);          //-> temp0
    const auto pool2Dims     = net->getPoolDims(encConv2Dims);                    //-> concat3

    const auto encConv3Dims  = net->getConvDims("enc_conv3", pool2Dims);          //-> temp0
    const auto pool3Dims     = net->getPoolDims(encConv3Dims);                    //-> concat4

    const auto encConv4Dims  = net->getConvDims("enc_conv4", pool3Dims);          //-> temp0
    const auto pool4Dims     = net->getPoolDims(encConv4Dims);                    //-> temp1

    const auto encConv5aDims = net->getConvDims("enc_conv5a", pool4Dims);         //-> temp0
    const auto encConv5bDims = net->getConvDims("enc_conv5b", encConv5aDims);     //-> temp1

    const auto upsample4Dims = net->getUpsampleDims(encConv5bDims);               //-> concat4
    const auto concat4Dims   = net->getConcatDims({upsample4Dims, pool3Dims});
    const auto decConv4aDims = net->getConvDims("dec_conv4a", concat4Dims);       //-> temp0
    const auto decConv4bDims = net->getConvDims("dec_conv4b", decConv4aDims);     //-> temp1

    const auto upsample3Dims = net->getUpsampleDims(decConv4bDims);               //-> concat3
    const auto concat3Dims   = net->getConcatDims({upsample3Dims, pool2Dims});
    const auto decConv3aDims = net->getConvDims("dec_conv3a", concat3Dims);       //-> temp0
    const auto decConv3bDims = net->getConvDims("dec_conv3b", decConv3aDims);     //-> temp1

    const auto upsample2Dims = net->getUpsampleDims(decConv3bDims);               //-> concat2
    const auto concat2Dims   = net->getConcatDims({upsample2Dims, pool1Dims});
    const auto decConv2aDims = net->getConvDims("dec_conv2a", concat2Dims);       //-> temp0
    const auto decConv2bDims = net->getConvDims("dec_conv2b", decConv2aDims);     //-> temp1

    const auto upsample1Dims = net->getUpsampleDims(decConv2bDims);               //-> concat1
    const auto concat1Dims   = net->getConcatDims({upsample1Dims, inputReorderDims});
    const auto decConv1aDims = net->getConvDims("dec_conv1a", concat1Dims);       //-> temp0
    const auto decConv1bDims = net->getConvDims("dec_conv1b", decConv1aDims);     //-> temp1

    const auto decConv0Dims  = net->getConvDims("dec_conv0", decConv1bDims);      //-> temp0

    const auto outputDims = TensorDims({3, tileH, tileW});

    // Allocate two temporary ping-pong buffers to decrease memory usage
    const auto temp0Dims = getMaxDims({
      encConv0Dims,
      encConv2Dims,
      encConv3Dims,
      encConv4Dims,
      encConv5aDims,
      decConv4aDims,
      decConv3aDims,
      decConv2aDims,
      decConv1aDims,
      decConv0Dims
    });

    const auto temp1Dims = getMaxDims({
      encConv1Dims,
      pool4Dims,
      encConv5bDims,
      decConv4bDims,
      decConv3bDims,
      decConv2bDims,
      decConv1bDims,
    });

    auto temp0 = net->newTensor(temp0Dims);
    auto temp1 = net->newTensor(temp1Dims);

    // Allocate enough memory to hold the concat outputs. Then use the first
    // half to hold the previous conv output and the second half to hold the
    // pool/orig image output. This works because everything is C dimension
    // outermost, padded to K floats, and all the concats are on the C dimension.
    auto concat1Dst = net->newTensor(concat1Dims);
    auto concat2Dst = net->newTensor(concat2Dims);
    auto concat3Dst = net->newTensor(concat3Dims);
    auto concat4Dst = net->newTensor(concat4Dims);

    auto concat1Src = net->getConcatSrc(concat1Dst, {upsample1Dims, inputReorderDims});
    auto concat2Src = net->getConcatSrc(concat2Dst, {upsample2Dims, pool1Dims});
    auto concat3Src = net->getConcatSrc(concat3Dst, {upsample3Dims, pool2Dims});
    auto concat4Src = net->getConcatSrc(concat4Dst, {upsample4Dims, pool3Dims});

    // Transfer function
    Ref<TransferFunction> transferFunc = makeTransferFunc();
    if (isnan(inputScale))
    {
      if (hdr)
        net->addAutoexposure(color, transferFunc);
      else
        transferFunc->setInputScale(1.f);
    }
    else
    {
      transferFunc->setInputScale(inputScale);
    }

    // Create the nodes
    inputReorder = net->addInputReorder(color, albedo, normal,
                                        concat1Src[1],
                                        transferFunc, hdr, snorm,
                                        alignment);

    auto encConv0 = net->addConv("enc_conv0", inputReorder->getDst(), temp0->view(encConv0Dims));

    auto encConv1 = net->addConv("enc_conv1", encConv0->getDst(), temp1->view(encConv1Dims));
    auto pool1    = net->addPool(encConv1->getDst(), concat2Src[1]);

    auto encConv2 = net->addConv("enc_conv2", pool1->getDst(), temp0->view(encConv2Dims));
    auto pool2    = net->addPool(encConv2->getDst(), concat3Src[1]);

    auto encConv3 = net->addConv("enc_conv3", pool2->getDst(), temp0->view(encConv3Dims));
    auto pool3    = net->addPool(encConv3->getDst(), concat4Src[1]);

    auto encConv4 = net->addConv("enc_conv4", pool3->getDst(), temp0->view(encConv4Dims));
    auto pool4    = net->addPool(encConv4->getDst(), temp1->view(pool4Dims));

    auto encConv5a = net->addConv("enc_conv5a", pool4->getDst(), temp0->view(encConv5aDims));
    auto encConv5b = net->addConv("enc_conv5b", encConv5a->getDst(), temp1->view(encConv5bDims));

    auto upsample4 = net->addUpsample(encConv5b->getDst(), concat4Src[0]);
    auto decConv4a = net->addConv("dec_conv4a", concat4Dst, temp0->view(decConv4aDims));
    auto decConv4b = net->addConv("dec_conv4b", decConv4a->getDst(), temp1->view(decConv4bDims));

    auto upsample3 = net->addUpsample(decConv4b->getDst(), concat3Src[0]);
    auto decConv3a = net->addConv("dec_conv3a", concat3Dst, temp0->view(decConv3aDims));
    auto decConv3b = net->addConv("dec_conv3b", decConv3a->getDst(), temp1->view(decConv3bDims));

    auto upsample2 = net->addUpsample(decConv3b->getDst(), concat2Src[0]);
    auto decConv2a = net->addConv("dec_conv2a", concat2Dst, temp0->view(decConv2aDims));
    auto decConv2b = net->addConv("dec_conv2b", decConv2a->getDst(), temp1->view(decConv2bDims));

    auto upsample1 = net->addUpsample(decConv2b->getDst(), concat1Src[0]);
    auto decConv1a = net->addConv("dec_conv1a", concat1Dst, temp0->view(decConv1aDims));
    auto decConv1b = net->addConv("dec_conv1b", decConv1a->getDst(), temp1->view(decConv1bDims));

    auto decConv0 = net->addConv("dec_conv0", decConv1b->getDst(), temp0->view(decConv0Dims), false);

    outputReorder = net->addOutputReorder(decConv0->getDst(),
                                          outputTemp ? outputTemp : output,
                                          transferFunc, hdr, snorm);

    net->finalize();
    return net;
  }

  // ---------------------------------------------------------------------------
  // RTFilter
  // ---------------------------------------------------------------------------

  RTFilter::RTFilter(const Ref<Device>& device)
    : UNetFilter(device)
  {
    builtinWeights.hdr           = blobs::weights::rt_hdr;
    builtinWeights.hdr_alb       = blobs::weights::rt_hdr_alb;
    builtinWeights.hdr_alb_nrm   = blobs::weights::rt_hdr_alb_nrm;
    builtinWeights.hdr_calb_cnrm = blobs::weights::rt_hdr_calb_cnrm;
    builtinWeights.ldr           = blobs::weights::rt_ldr;
    builtinWeights.ldr_alb       = blobs::weights::rt_ldr_alb;
    builtinWeights.ldr_alb_nrm   = blobs::weights::rt_ldr_alb_nrm;
  }

  Ref<TransferFunction> RTFilter::makeTransferFunc()
  {
    if (srgb || !color)
      return makeRef<TransferFunction>(TransferFunction::Type::Linear);
    else if (hdr)
      return makeRef<TransferFunction>(TransferFunction::Type::PU);
    else
      return makeRef<TransferFunction>(TransferFunction::Type::SRGB);
  }

  void RTFilter::setImage(const std::string& name, const Image& data)
  {
    if (name == "color")
      color = data;
    else if (name == "albedo")
      albedo = data;
    else if (name == "normal")
      normal = data;
    else if (name == "output")
      output = data;
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTFilter::set1i(const std::string& name, int value)
  {
    if (name == "hdr")
      hdr = value;
    else if (name == "srgb")
      srgb = value;
    else if (name == "cleanAux")
      cleanAux = value;
    else if (name == "maxMemoryMB")
      maxMemoryMB = value;
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  int RTFilter::get1i(const std::string& name)
  {
    if (name == "hdr")
      return hdr;
    else if (name == "srgb")
      return srgb;
    else if (name == "cleanAux")
      return cleanAux;
    else if (name == "maxMemoryMB")
      return maxMemoryMB;
    else if (name == "alignment")
      return alignment;
    else if (name == "overlap")
      return overlap;
    else
      throw Exception(Error::InvalidArgument, "unknown filter parameter");
  }

  // ---------------------------------------------------------------------------
  // RTLightmapFilter
  // ---------------------------------------------------------------------------

  RTLightmapFilter::RTLightmapFilter(const Ref<Device>& device)
    : UNetFilter(device)
  {
    hdr = true;

    builtinWeights.hdr = blobs::weights::rtlightmap_hdr;
    builtinWeights.dir = blobs::weights::rtlightmap_dir;
  }

  Ref<TransferFunction> RTLightmapFilter::makeTransferFunc()
  {
    if (hdr)
      return makeRef<TransferFunction>(TransferFunction::Type::Log);
    else
      return makeRef<TransferFunction>(TransferFunction::Type::Linear);
  }

  void RTLightmapFilter::setImage(const std::string& name, const Image& data)
  {
    if (name == "color")
      color = data;
    else if (name == "output")
      output = data;
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTLightmapFilter::set1i(const std::string& name, int value)
  {
    if (name == "directional")
    {
      directional = value;
      hdr = !directional;
    }
    else if (name == "maxMemoryMB")
      maxMemoryMB = value;
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  int RTLightmapFilter::get1i(const std::string& name)
  {
    if (name == "directional")
      return directional;
    else if (name == "maxMemoryMB")
      return maxMemoryMB;
    else if (name == "alignment")
      return alignment;
    else if (name == "overlap")
      return overlap;
    else
      throw Exception(Error::InvalidArgument, "unknown filter parameter");
  }

} // namespace oidn
