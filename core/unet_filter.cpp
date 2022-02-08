// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "tza.h"
#include "unet_filter.h"

namespace oidn {

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
      device->runTask([&]()
      {
        init();
      });

      device->wait();
    }

    dirty = false;
    dirtyParam = false;
  }

  void UNetFilter::execute(bool sync)
  {
    if (dirty)
      throw Exception(Error::InvalidOperation, "changes to the filter are not committed");

    if (H <= 0 || W <= 0)
      return;

    device->runTask([&]()
    {
      // Initialize the progress state
      double workAmount = tileCountH * tileCountW * net->getWorkAmount();
      if (outputTemp)
        workAmount += 1;
      Progress progress(progressFunc, progressUserPtr, workAmount);

      // Set the input and output
      inputProcess->setSrc(color, albedo, normal);
      outputProcess->setDst(outputTemp ? outputTemp : output);

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
          inputProcess->setTile(h, w,
                           alignOffsetH, alignOffsetW,
                           tileH1, tileW1);

          // Set the output tile
          outputProcess->setTile(alignOffsetH + overlapBeginH, alignOffsetW + overlapBeginW,
                            h + overlapBeginH, w + overlapBeginW,
                            tileH2, tileW2);

          //printf("Tile: %d %d -> %d %d\n", w+overlapBeginW, h+overlapBeginH, w+overlapBeginW+tileW2, h+overlapBeginH+tileH2);

          // Denoise the tile
          net->run(progress);

          // Next tile
          tileIndex++;
        }
      }

      // Copy the output image to the final buffer if filtering in-place
      if (outputTemp)
        device->imageCopy(*outputTemp, *output);

      // Finished
      progress.finish();
    });

    if (sync)
      device->wait();
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
    inputProcess = nullptr;
    outputProcess = nullptr;
    transferFunc = nullptr;
    outputTemp = nullptr;

    // Check the input/output buffers
    if (!color && !albedo && !normal)
      throw Exception(Error::InvalidOperation, "input image not specified");
    if (!output)
      throw Exception(Error::InvalidOperation, "output image not specified");

    H = output->getH();
    W = output->getW();

    if (((color  && color->getFormat()  != Format::Float3) ||
         (albedo && albedo->getFormat() != Format::Float3) ||
         (normal && normal->getFormat() != Format::Float3)) &&
        ((color  && color->getFormat()  != Format::Half3) ||
         (albedo && albedo->getFormat() != Format::Half3) ||
         (normal && normal->getFormat() != Format::Half3)))
      throw Exception(Error::InvalidOperation, "unsupported input image format");

    if (output->getFormat() != Format::Float3 && output->getFormat() != Format::Half3)
      throw Exception(Error::InvalidOperation, "unsupported output image format");

    if ((color  && (color->getW()  != W || color->getH()  != H)) ||
        (albedo && (albedo->getW() != W || albedo->getH() != H)) ||
        (normal && (normal->getW() != W || normal->getH() != H)))
      throw Exception(Error::InvalidOperation, "image size mismatch");

    if (directional && (hdr || srgb))
      throw Exception(Error::InvalidOperation, "directional and hdr/srgb modes cannot be enabled at the same time");
    if (hdr && srgb)
      throw Exception(Error::InvalidOperation, "hdr and srgb modes cannot be enabled at the same time");

    if (device->isVerbose(2))
    {
      std::cout << "Inputs:";
      if (color)  std::cout << " " << (directional ? "dir" : (hdr ? "hdr" : "ldr")) << ":" << color->getFormat();
      if (albedo) std::cout << " " << "alb" << ":" << albedo->getFormat();
      if (normal) std::cout << " " << "nrm" << ":" << normal->getFormat();
      std::cout << std::endl;
      std::cout << "Output: " << output->getFormat() << std::endl;
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
    TensorDesc inputDesc = net->getInputDesc({inputC, tileH, tileW}, alignment);

    TensorDesc encConv0Desc  = net->getConvDesc("enc_conv0", inputDesc);

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
    TensorDesc concat1Desc   = net->getConcatDesc({upsample1Desc, inputDesc});
    TensorDesc decConv1aDesc = net->getConvDesc("dec_conv1a", concat1Desc);
    TensorDesc decConv1bDesc = net->getConvDesc("dec_conv1b", decConv1aDesc);

    TensorDesc decConv0Desc  = net->getConvDesc("dec_conv0", decConv1bDesc);

    // Compute the tensor offsets
    ptrdiff_t endOfs = 0; // we'll have negative offsets relative to the end of the buffer
    ptrdiff_t inputOfs     = endOfs - inputDesc.getAlignedSize();
    ptrdiff_t encConv0Ofs  = inputOfs - encConv0Desc.getAlignedSize();
    ptrdiff_t pool1Ofs     = inputOfs - pool1Desc.getAlignedSize();
    ptrdiff_t encConv1Ofs  = min(encConv0Ofs, pool1Ofs) - encConv1Desc.getAlignedSize();
    ptrdiff_t pool2Ofs     = pool1Ofs - pool2Desc.getAlignedSize();
    ptrdiff_t encConv2Ofs  = pool2Ofs - encConv2Desc.getAlignedSize();
    ptrdiff_t pool3Ofs     = pool2Ofs - pool3Desc.getAlignedSize();
    ptrdiff_t encConv3Ofs  = pool3Ofs - encConv3Desc.getAlignedSize();
    ptrdiff_t encConv4Ofs  = pool3Ofs - encConv4Desc.getAlignedSize();
    ptrdiff_t encConv5aOfs = pool3Ofs - encConv5aDesc.getAlignedSize();
    ptrdiff_t pool4Ofs     = min(encConv4Ofs, encConv5aOfs) - pool4Desc.getAlignedSize();
    ptrdiff_t upsample4Ofs = pool3Ofs - upsample4Desc.getAlignedSize();
    ptrdiff_t encConv5bOfs = min(encConv5aOfs, upsample4Ofs) - encConv5bDesc.getAlignedSize();
    ptrdiff_t upsample3Ofs = pool2Ofs - upsample3Desc.getAlignedSize();
    ptrdiff_t decConv4bOfs = upsample3Ofs - decConv4bDesc.getAlignedSize();
    ptrdiff_t decConv4aOfs = min(upsample4Ofs, decConv4bOfs) - decConv4aDesc.getAlignedSize();
    ptrdiff_t upsample2Ofs = pool1Ofs - upsample2Desc.getAlignedSize();
    ptrdiff_t decConv3bOfs = upsample2Ofs - decConv3bDesc.getAlignedSize();
    ptrdiff_t decConv3aOfs = min(upsample3Ofs, decConv3bOfs) - decConv3aDesc.getAlignedSize();
    ptrdiff_t upsample1Ofs = inputOfs - upsample1Desc.getAlignedSize();
    ptrdiff_t decConv2bOfs = upsample1Ofs - decConv2bDesc.getAlignedSize();
    ptrdiff_t decConv2aOfs = min(upsample2Ofs, decConv2bOfs) - decConv2aDesc.getAlignedSize();
    ptrdiff_t decConv1bOfs = endOfs - decConv1bDesc.getAlignedSize();
    ptrdiff_t decConv1aOfs = min(upsample1Ofs, decConv1bOfs) - decConv1aDesc.getAlignedSize();
    ptrdiff_t decConv0Ofs  = decConv1bOfs - decConv0Desc.getAlignedSize();

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
    ImageDesc outputTempDesc(output->getFormat(), W, H);
    ptrdiff_t outputTempOfs = 0;
    if (inplace && (tileCountH * tileCountW) > 1)
    {
      outputTempOfs = minOfs - outputTempDesc.getAlignedSize();
      minOfs = outputTempOfs;
    }

    // Compute the size of the scratch buffer
    const size_t scratchSize = -minOfs;
    if (getScratchSizeOnly)
      return scratchSize;

    // Allocate the scratch buffer
    net->allocScratch(scratchSize);

    // Create the tensors in the scratch buffer
    auto input     = net->newTensor(inputDesc, inputOfs);
    auto encConv0  = net->newTensor(encConv0Desc, encConv0Ofs);
    auto encConv1  = net->newTensor(encConv1Desc, encConv1Ofs);
    auto pool1     = net->newTensor(pool1Desc, pool1Ofs);
    auto encConv2  = net->newTensor(encConv2Desc, encConv2Ofs);
    auto pool2     = net->newTensor(pool2Desc, pool2Ofs);
    auto encConv3  = net->newTensor(encConv3Desc, encConv3Ofs);
    auto pool3     = net->newTensor(pool3Desc, pool3Ofs);
    auto encConv4  = net->newTensor(encConv4Desc, encConv4Ofs);
    auto pool4     = net->newTensor(pool4Desc, pool4Ofs);
    auto encConv5a = net->newTensor(encConv5aDesc, encConv5aOfs);
    auto encConv5b = net->newTensor(encConv5bDesc, encConv5bOfs);
    auto upsample4 = net->newTensor(upsample4Desc, upsample4Ofs);
    auto decConv4a = net->newTensor(decConv4aDesc, decConv4aOfs);
    auto decConv4b = net->newTensor(decConv4bDesc, decConv4bOfs);
    auto upsample3 = net->newTensor(upsample3Desc, upsample3Ofs);
    auto decConv3a = net->newTensor(decConv3aDesc, decConv3aOfs);
    auto decConv3b = net->newTensor(decConv3bDesc, decConv3bOfs);
    auto upsample2 = net->newTensor(upsample2Desc, upsample2Ofs);
    auto decConv2a = net->newTensor(decConv2aDesc, decConv2aOfs);
    auto decConv2b = net->newTensor(decConv2bDesc, decConv2bOfs);
    auto upsample1 = net->newTensor(upsample1Desc, upsample1Ofs);
    auto decConv1a = net->newTensor(decConv1aDesc, decConv1aOfs);
    auto decConv1b = net->newTensor(decConv1bDesc, decConv1bOfs);
    auto decConv0  = net->newTensor(decConv0Desc, decConv0Ofs);

    // Create the ops
    const bool snorm = directional || (!color && normal);

    transferFunc = getTransferFunc();
    inputProcess = net->addInputProcess("input", input, transferFunc, hdr, snorm);

    net->addConv("enc_conv0", input, encConv0);

    net->addConv("enc_conv1", encConv0, encConv1);
    net->addPool("pool1", encConv1, pool1);

    net->addConv("enc_conv2", pool1, encConv2);
    net->addPool("pool2", encConv2, pool2);

    net->addConv("enc_conv3", pool2, encConv3);
    net->addPool("pool3", encConv3, pool3);

    net->addConv("enc_conv4", pool3, encConv4);
    net->addPool("pool4", encConv4, pool4);

    net->addConv("enc_conv5a", pool4, encConv5a);
    net->addConv("enc_conv5b", encConv5a, encConv5b);

    net->addUpsample("upsample4", encConv5b, upsample4);
    net->addConcatConv("dec_conv4a", upsample4, pool3, decConv4a);
    net->addConv("dec_conv4b", decConv4a, decConv4b);

    net->addUpsample("upsample3", decConv4b, upsample3);
    net->addConcatConv("dec_conv3a", upsample3, pool2, decConv3a);
    net->addConv("dec_conv3b", decConv3a, decConv3b);

    net->addUpsample("upsample2", decConv3b, upsample2);
    net->addConcatConv("dec_conv2a", upsample2, pool1, decConv2a);
    net->addConv("dec_conv2b", decConv2a, decConv2b);

    net->addUpsample("upsample1", decConv2b, upsample1);
    net->addConcatConv("dec_conv1a", upsample1, input, decConv1a);
    net->addConv("dec_conv1b", decConv1a, decConv1b);
    
    net->addConv("dec_conv0", decConv1b, decConv0, false);

    outputProcess = net->addOutputProcess("output", decConv0, transferFunc, hdr, snorm);

    // Create the temporary output
    if (outputTempOfs)
      outputTemp = net->newImage(outputTempDesc, outputTempOfs);

    // Finalize the network
    net->finalize();

    return scratchSize;
  }

} // namespace oidn
