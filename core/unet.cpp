// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tza.h"
#include "output_copy.h"
#include "unet.h"

// Default weights
#if defined(OIDN_FILTER_RT)
  #include "weights/rt_hdr.h"
  #include "weights/rt_hdr_alb.h"
  #include "weights/rt_hdr_alb_nrm.h"
  #include "weights/rt_hdr_calb_cnrm.h"
  #include "weights/rt_ldr.h"
  #include "weights/rt_ldr_alb.h"
  #include "weights/rt_ldr_alb_nrm.h"
  #include "weights/rt_ldr_calb_cnrm.h"
  #include "weights/rt_alb.h"
  #include "weights/rt_nrm.h"
#endif

#if defined(OIDN_FILTER_RTLIGHTMAP)
  #include "weights/rtlightmap_hdr.h"
  #include "weights/rtlightmap_dir.h"
#endif

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
      setParam(userWeights, data);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void UNetFilter::updateData(const std::string& name)
  {
    if (name == "weights")
      dirtyParam |= userWeights;
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void UNetFilter::removeData(const std::string& name)
  {
    if (name == "weights")
      removeParam(userWeights);
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

    // Determine whether in-place filtering is required
    bool inplaceNew = output &&
                      ((color  && output->overlaps(*color))  ||
                       (albedo && output->overlaps(*albedo)) ||
                       (normal && output->overlaps(*normal)));
    setParam(inplace, inplaceNew);

    if (dirtyParam)
    {
      // (Re-)Initialize the filter
      device->executeTask([&]()
      {
        init();
      });
    }

    dirty = false;
    dirtyParam = false;
  }

  void UNetFilter::execute()
  {
    if (dirty)
      throw Exception(Error::InvalidOperation, "changes to the filter are not committed");

    if (H <= 0 || W <= 0)
      return;

    device->executeTask([&]()
    {
      // Initialize the progress state
      double workAmount = tileCountH * tileCountW * net->getWorkAmount();
      if (outputTemp)
        workAmount += 1;
      Progress progress(progressFunc, progressUserPtr, workAmount);

      // Set the input and output
      inputReorder->setSrc(color, albedo, normal);
      outputReorder->setDst(outputTemp ? outputTemp : output);

      // Set the input scale
      if (isnan(inputScale))
      {
        if (hdr)
          transferFunc->setInputScale(getAutoexposure(*color));
        else
          transferFunc->setInputScale(1.f);
      }
      else
      {
        transferFunc->setInputScale(inputScale);
      }

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
        outputCopy(*outputTemp, *output);

      // Finished
      progress.finish();
    });
  }

  void UNetFilter::computeTileSize()
  {
    const int minTileSize = 3*overlap;

    // Compute the maximum allowed scratch size to fit into the requested memory limit
    const size_t maxScratchSize = size_t(maxMemoryMB)*1024*1024;

    tileCountH = 1;
    tileCountW = 1;
    tileH = round_up(H, alignment);
    tileW = round_up(W, alignment);

    // Divide the image into tiles until the scratch size gets below the threshold
    while (buildNet(true) > maxScratchSize)
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

  void UNetFilter::init()
  {
    // Cleanup
    net = nullptr;
    inputReorder = nullptr;
    outputReorder = nullptr;
    transferFunc = nullptr;
    outputTemp = nullptr;

    // Check the input/output buffers
    if (!color && !albedo && !normal)
      throw Exception(Error::InvalidOperation, "input image not specified");
    if (!output)
      throw Exception(Error::InvalidOperation, "output image not specified");

    H = output->height;
    W = output->width;

    if ((color  && color->format  != Format::Float3) ||
        (albedo && albedo->format != Format::Float3) ||
        (normal && normal->format != Format::Float3) ||
        (output->format != Format::Float3))
      throw Exception(Error::InvalidOperation, "unsupported image format");

    if ((color  && (color->width  != W || color->height  != H)) ||
        (albedo && (albedo->width != W || albedo->height != H)) ||
        (normal && (normal->width != W || normal->height != H)))
      throw Exception(Error::InvalidOperation, "image size mismatch");

    if (directional && (hdr || srgb))
      throw Exception(Error::InvalidOperation, "directional and hdr/srgb modes cannot be enabled at the same time");
    if (hdr && srgb)
      throw Exception(Error::InvalidOperation, "hdr and srgb modes cannot be enabled at the same time");

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

    if (color)
    {
      if (!albedo && !normal)
      {
        weights = directional ? defaultWeights.dir : (hdr ? defaultWeights.hdr : defaultWeights.ldr);
      }
      else if (albedo && !normal)
      {
        weights = hdr ? defaultWeights.hdr_alb : defaultWeights.ldr_alb;
      }
      else if (albedo && normal)
      {
        if (cleanAux)
          weights = hdr ? defaultWeights.hdr_calb_cnrm : defaultWeights.ldr_calb_cnrm;
        else
          weights = hdr ? defaultWeights.hdr_alb_nrm : defaultWeights.ldr_alb_nrm;
      }
    }
    else
    {
      // Auxiliary feature filtering
      if (albedo && !normal)
      {
        if (hdr)
          throw Exception(Error::InvalidOperation, "hdr mode is not supported for albedo filtering");
        weights = defaultWeights.alb;
      }
      else if (!albedo && normal)
      {
        if (hdr || srgb)
          throw Exception(Error::InvalidOperation, "hdr and srgb modes are not supported for normal filtering");
        weights = defaultWeights.nrm;
      }
      else
      {
        throw Exception(Error::InvalidOperation, "invalid combination of input features");
      }
    }

    if (userWeights)
      weights = userWeights;

    if (!weights)
      throw Exception(Error::InvalidOperation, "unsupported combination of input features");

    // Parse the weights blob
    const auto weightsMap = parseTZA(device, weights.ptr, weights.size);

    // Create the network
    net.reset(new Network(device, weightsMap));

    // Compute the tile size
    computeTileSize();

    // If the image size is zero, there is nothing else to do
    if (H <= 0 || W <= 0)
      return;

    // Build the network
    buildNet();
  }

  // Builds the network (optional) and returns the size of the scratch memory
  size_t UNetFilter::buildNet(bool getScratchSizeOnly)
  {
    // If the image size is zero, there is nothing else to do
    if (H <= 0 || W <= 0)
      return 0;

    // Get the number of input channels
    int inputC = 0;
    if (color)  inputC += 3;
    if (albedo) inputC += 3;
    if (normal) inputC += 3;

    // Compute the tensor descriptors
    TensorDims inputDims = TensorDims({inputC, tileH, tileW});

    TensorDesc inputReorderDesc = net->getInputReorderDesc(inputDims, alignment);

    TensorDesc encConv0Desc  = net->getConvDesc("enc_conv0", inputReorderDesc);

    TensorDesc encConv1Desc  = net->getConvDesc("enc_conv1", encConv0Desc);
    TensorDesc pool1Desc     = net->getPoolDesc(encConv1Desc);

    TensorDesc encConv2Desc  = net->getConvDesc("enc_conv2", pool1Desc);
    TensorDesc pool2Desc     = net->getPoolDesc(encConv2Desc);

    TensorDesc encConv3Desc  = net->getConvDesc("enc_conv3", pool2Desc);
    TensorDesc pool3Desc     = net->getPoolDesc(encConv3Desc);

    TensorDesc encConv4Desc  = net->getConvDesc("enc_conv4", pool3Desc);
    TensorDesc pool4Desc     = net->getPoolDesc(encConv4Desc);

    TensorDesc encConv5aDesc = net->getConvDesc("enc_conv5a", pool4Desc);
    TensorDesc encConv5bDesc = net->getConvDesc("enc_conv5b", encConv5aDesc);

    TensorDesc upsample4Desc = net->getUpsampleDesc(encConv5bDesc);
    TensorDesc concat4Desc   = net->getConcatDesc({upsample4Desc, pool3Desc});
    TensorDesc decConv4aDesc = net->getConvDesc("dec_conv4a", concat4Desc);
    TensorDesc decConv4bDesc = net->getConvDesc("dec_conv4b", decConv4aDesc);

    TensorDesc upsample3Desc = net->getUpsampleDesc(decConv4bDesc);
    TensorDesc concat3Desc   = net->getConcatDesc({upsample3Desc, pool2Desc});
    TensorDesc decConv3aDesc = net->getConvDesc("dec_conv3a", concat3Desc);
    TensorDesc decConv3bDesc = net->getConvDesc("dec_conv3b", decConv3aDesc);

    TensorDesc upsample2Desc = net->getUpsampleDesc(decConv3bDesc);
    TensorDesc concat2Desc   = net->getConcatDesc({upsample2Desc, pool1Desc});
    TensorDesc decConv2aDesc = net->getConvDesc("dec_conv2a", concat2Desc);
    TensorDesc decConv2bDesc = net->getConvDesc("dec_conv2b", decConv2aDesc);

    TensorDesc upsample1Desc = net->getUpsampleDesc(decConv2bDesc);
    TensorDesc concat1Desc   = net->getConcatDesc({upsample1Desc, inputReorderDesc});
    TensorDesc decConv1aDesc = net->getConvDesc("dec_conv1a", concat1Desc);
    TensorDesc decConv1bDesc = net->getConvDesc("dec_conv1b", decConv1aDesc);

    TensorDesc decConv0Desc  = net->getConvDesc("dec_conv0", decConv1bDesc);

    // Compute the tensor offsets
    ptrdiff_t endOfs = 0; // we'll have negative offsets relative to the end of the buffer
    ptrdiff_t inputReorderOfs = endOfs - inputReorderDesc.alignedByteSize();
    ptrdiff_t encConv0Ofs  = inputReorderOfs - encConv0Desc.alignedByteSize();
    ptrdiff_t pool1Ofs     = inputReorderOfs - pool1Desc.alignedByteSize();
    ptrdiff_t encConv1Ofs  = min(encConv0Ofs, pool1Ofs) - encConv1Desc.alignedByteSize();
    ptrdiff_t pool2Ofs     = pool1Ofs - pool2Desc.alignedByteSize();
    ptrdiff_t encConv2Ofs  = pool2Ofs - encConv2Desc.alignedByteSize();
    ptrdiff_t pool3Ofs     = pool2Ofs - pool3Desc.alignedByteSize();
    ptrdiff_t encConv3Ofs  = pool3Ofs - encConv3Desc.alignedByteSize();
    ptrdiff_t encConv4Ofs  = pool3Ofs - encConv4Desc.alignedByteSize();
    ptrdiff_t encConv5aOfs = pool3Ofs - encConv5aDesc.alignedByteSize();
    ptrdiff_t pool4Ofs     = min(encConv4Ofs, encConv5aOfs) - pool4Desc.alignedByteSize();
    ptrdiff_t upsample4Ofs = pool3Ofs - upsample4Desc.alignedByteSize();
    ptrdiff_t encConv5bOfs = min(encConv5aOfs, upsample4Ofs) - encConv5bDesc.alignedByteSize();
    ptrdiff_t upsample3Ofs = pool2Ofs - upsample3Desc.alignedByteSize();
    ptrdiff_t decConv4bOfs = upsample3Ofs - decConv4bDesc.alignedByteSize();
    ptrdiff_t decConv4aOfs = min(upsample4Ofs, decConv4bOfs) - decConv4aDesc.alignedByteSize();
    ptrdiff_t upsample2Ofs = pool1Ofs - upsample2Desc.alignedByteSize();
    ptrdiff_t decConv3bOfs = upsample2Ofs - decConv3bDesc.alignedByteSize();
    ptrdiff_t decConv3aOfs = min(upsample3Ofs, decConv3bOfs) - decConv3aDesc.alignedByteSize();
    ptrdiff_t upsample1Ofs = inputReorderOfs - upsample1Desc.alignedByteSize();
    ptrdiff_t decConv2bOfs = upsample1Ofs - decConv2bDesc.alignedByteSize();
    ptrdiff_t decConv2aOfs = min(upsample2Ofs, decConv2bOfs) - decConv2aDesc.alignedByteSize();
    ptrdiff_t decConv1bOfs = endOfs - decConv1bDesc.alignedByteSize();
    ptrdiff_t decConv1aOfs = min(upsample1Ofs, decConv1bOfs) - decConv1aDesc.alignedByteSize();
    ptrdiff_t decConv0Ofs  = decConv1bOfs - decConv0Desc.alignedByteSize();

    const std::vector<ptrdiff_t> minOfsList = {
      encConv1Ofs,
      encConv2Ofs,
      encConv3Ofs,
      pool4Ofs,
      encConv5bOfs,
      decConv4aOfs,
      decConv3aOfs,
      decConv2aOfs,
      decConv1aOfs,
      decConv0Ofs
    };
    ptrdiff_t minOfs = *std::min_element(minOfsList.begin(), minOfsList.end());

    // If doing in-place _tiled_ filtering, we need a temporary output buffer too
    ImageDesc outputTempDesc(output->format, W, H);
    ptrdiff_t outputTempOfs = 0;
    if (inplace && (tileCountH * tileCountW) > 1)
    {
      outputTempOfs = minOfs - outputTempDesc.alignedByteSize();
      minOfs = outputTempOfs;
    }

    // Compute the size of the scratch buffer
    const size_t scratchSize = -minOfs;
    if (getScratchSizeOnly)
      return scratchSize;

    // Allocate the scratch buffer
    net->allocScratch(scratchSize);

    // Create the transfer function
    transferFunc = getTransferFunc();

    // Create the nodes
    const bool snorm = directional || (!color && normal);

    inputReorder = net->addInputReorder(net->newTensor(inputReorderDesc, inputReorderOfs),
                                        transferFunc, hdr, snorm);

    auto encConv0 = net->addConv("enc_conv0",
                                 inputReorder->getDst(),
                                 net->newTensor(encConv0Desc, encConv0Ofs));

    auto encConv1 = net->addConv("enc_conv1",
                                 encConv0->getDst(),
                                 net->newTensor(encConv1Desc, encConv1Ofs));

    auto pool1 = net->addPool(encConv1->getDst(),
                              net->newTensor(pool1Desc, pool1Ofs));

    auto encConv2 = net->addConv("enc_conv2",
                                 pool1->getDst(),
                                 net->newTensor(encConv2Desc, encConv2Ofs));

    auto pool2 = net->addPool(encConv2->getDst(),
                              net->newTensor(pool2Desc, pool2Ofs));

    auto encConv3 = net->addConv("enc_conv3",
                                 pool2->getDst(),
                                 net->newTensor(encConv3Desc, encConv3Ofs));

    auto pool3 = net->addPool(encConv3->getDst(),
                              net->newTensor(pool3Desc, pool3Ofs));

    auto encConv4 = net->addConv("enc_conv4",
                                 pool3->getDst(),
                                 net->newTensor(encConv4Desc, encConv4Ofs));

    auto pool4 = net->addPool(encConv4->getDst(),
                              net->newTensor(pool4Desc, pool4Ofs));

    auto encConv5a = net->addConv("enc_conv5a",
                                  pool4->getDst(),
                                  net->newTensor(encConv5aDesc, encConv5aOfs));

    auto encConv5b = net->addConv("enc_conv5b",
                                  encConv5a->getDst(),
                                  net->newTensor(encConv5bDesc, encConv5bOfs));

    auto upsample4 = net->addUpsample(encConv5b->getDst(),
                                      net->newTensor(upsample4Desc, upsample4Ofs));

    auto decConv4a = net->addConv("dec_conv4a",
                                  net->newTensor(concat4Desc, upsample4Ofs),
                                  net->newTensor(decConv4aDesc, decConv4aOfs));

    auto decConv4b = net->addConv("dec_conv4b",
                                  decConv4a->getDst(),
                                  net->newTensor(decConv4bDesc, decConv4bOfs));

    auto upsample3 = net->addUpsample(decConv4b->getDst(),
                                      net->newTensor(upsample3Desc, upsample3Ofs));

    auto decConv3a = net->addConv("dec_conv3a",
                                  net->newTensor(concat3Desc, upsample3Ofs),
                                  net->newTensor(decConv3aDesc, decConv3aOfs));

    auto decConv3b = net->addConv("dec_conv3b",
                                  decConv3a->getDst(),
                                  net->newTensor(decConv3bDesc, decConv3bOfs));

    auto upsample2 = net->addUpsample(decConv3b->getDst(),
                                      net->newTensor(upsample2Desc, upsample2Ofs));

    auto decConv2a = net->addConv("dec_conv2a",
                                  net->newTensor(concat2Desc, upsample2Ofs),
                                  net->newTensor(decConv2aDesc, decConv2aOfs));

    auto decConv2b = net->addConv("dec_conv2b",
                                  decConv2a->getDst(),
                                  net->newTensor(decConv2bDesc, decConv2bOfs));

    auto upsample1 = net->addUpsample(decConv2b->getDst(),
                                      net->newTensor(upsample1Desc, upsample1Ofs));

    auto decConv1a = net->addConv("dec_conv1a",
                                  net->newTensor(concat1Desc, upsample1Ofs),
                                  net->newTensor(decConv1aDesc, decConv1aOfs));

    auto decConv1b = net->addConv("dec_conv1b",
                                  decConv1a->getDst(),
                                  net->newTensor(decConv1bDesc, decConv1bOfs));

    auto decConv0 = net->addConv("dec_conv0",
                                 decConv1b->getDst(),
                                 net->newTensor(decConv0Desc, decConv0Ofs),
                                 false);

    outputReorder = net->addOutputReorder(decConv0->getDst(),
                                          transferFunc, hdr, snorm);

    // Create the temporary output
    if (outputTempOfs)
      outputTemp = net->newImage(outputTempDesc, outputTempOfs);

    // Finalize the network
    net->finalize();

    return scratchSize;
  }

  // ---------------------------------------------------------------------------
  // RTFilter
  // ---------------------------------------------------------------------------

  RTFilter::RTFilter(const Ref<Device>& device)
    : UNetFilter(device)
  {
  #if defined(OIDN_FILTER_RT)
    defaultWeights.hdr           = blobs::weights::rt_hdr;
    defaultWeights.hdr_alb       = blobs::weights::rt_hdr_alb;
    defaultWeights.hdr_alb_nrm   = blobs::weights::rt_hdr_alb_nrm;
    defaultWeights.hdr_calb_cnrm = blobs::weights::rt_hdr_calb_cnrm;
    defaultWeights.ldr           = blobs::weights::rt_ldr;
    defaultWeights.ldr_alb       = blobs::weights::rt_ldr_alb;
    defaultWeights.ldr_alb_nrm   = blobs::weights::rt_ldr_alb_nrm;
    defaultWeights.ldr_calb_cnrm = blobs::weights::rt_ldr_calb_cnrm;
    defaultWeights.alb           = blobs::weights::rt_alb;
    defaultWeights.nrm           = blobs::weights::rt_nrm;
  #endif
  }

  std::shared_ptr<TransferFunction> RTFilter::getTransferFunc()
  {
    if (srgb || (!color && normal))
      return std::make_shared<TransferFunction>(TransferFunction::Type::Linear);
    else if (hdr)
      return std::make_shared<TransferFunction>(TransferFunction::Type::PU);
    else
      return std::make_shared<TransferFunction>(TransferFunction::Type::SRGB);
  }

  void RTFilter::setImage(const std::string& name, const std::shared_ptr<Image>& image)
  {
    if (name == "color")
      setParam(color, image);
    else if (name == "albedo")
      setParam(albedo, image);
    else if (name == "normal")
      setParam(normal, image);
    else if (name == "output")
      setParam(output, image);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTFilter::removeImage(const std::string& name)
  {
    if (name == "color")
      removeParam(color);
    else if (name == "albedo")
      removeParam(albedo);
    else if (name == "normal")
      removeParam(normal);
    else if (name == "output")
      removeParam(output);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTFilter::set1i(const std::string& name, int value)
  {
    if (name == "hdr")
      setParam(hdr, value);
    else if (name == "srgb")
      setParam(srgb, value);
    else if (name == "cleanAux")
      setParam(cleanAux, value);
    else if (name == "maxMemoryMB")
      setParam(maxMemoryMB, value);
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

  #if defined(OIDN_FILTER_RTLIGHTMAP)
    defaultWeights.hdr = blobs::weights::rtlightmap_hdr;
    defaultWeights.dir = blobs::weights::rtlightmap_dir;
  #endif
  }

  std::shared_ptr<TransferFunction> RTLightmapFilter::getTransferFunc()
  {
    if (hdr)
      return std::make_shared<TransferFunction>(TransferFunction::Type::Log);
    else
      return std::make_shared<TransferFunction>(TransferFunction::Type::Linear);
  }

  void RTLightmapFilter::setImage(const std::string& name, const std::shared_ptr<Image>& image)
  {
    if (name == "color")
      setParam(color, image);
    else if (name == "output")
      setParam(output, image);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTLightmapFilter::removeImage(const std::string& name)
  {
    if (name == "color")
      removeParam(color);
    else if (name == "output")
      removeParam(output);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void RTLightmapFilter::set1i(const std::string& name, int value)
  {
    if (name == "directional")
    {
      setParam(directional, value);
      hdr = !directional;
    }
    else if (name == "maxMemoryMB")
      setParam(maxMemoryMB, value);
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
