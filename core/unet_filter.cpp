// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "unet_filter.h"
#include "autoexposure.h"

namespace oidn {

  UNetFilter::UNetFilter(const Ref<Device>& device)
    : Filter(device),
      progress(device)
  {
    maxMemoryMB = int(600 * getDataTypeSize(device->getTensorDataType()));
  }
  
  UNetFilter::~UNetFilter()
  {
    // Make sure that all asynchronous operations have completed
    device->wait();
  }

  void UNetFilter::setData(const std::string& name, const Data& data)
  {
    if (name == "weights")
      setParam(userWeightsBlob, data);
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void UNetFilter::updateData(const std::string& name)
  {
    if (name == "weights")
      dirtyParam |= userWeightsBlob;
    else
      device->warning("unknown filter parameter");

    dirty = true;
  }

  void UNetFilter::removeData(const std::string& name)
  {
    if (name == "weights")
      removeParam(userWeightsBlob);
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
      // Make sure that all asynchronous operations have completed
      device->wait();

      // (Re-)Initialize the filter
      device->runHostTask([&]() { init(); });
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

    device->runHostTask([&]()
    {
      // Initialize the progress state
      double workAmount = tileCountH * tileCountW * net->getWorkAmount();
      if (hdr && math::isnan(inputScale))
        workAmount += 1;
      if (outputTemp)
        workAmount += 1;
      progress.start(progressFunc, progressUserPtr, workAmount);

      // Set the input scale
      if (math::isnan(inputScale))
      {
        if (hdr)
        {
          autoexposure->setSrc(color);
          autoexposure->run();
          progress.update(1);
          /*
          float autoexpResult;
          device->memcpy(&autoexpResult, autoexposure->getResult(), sizeof(float));
          std::cout << "Autoexposure: " << autoexpResult << std::endl;
          */
          transferFunc->setInputScale(autoexposure->getResult());
        }
        else
        {
          transferFunc->setInputScale(1);
        }
      }
      else
      {
        transferFunc->setInputScale(inputScale);
      }

      // Set the input and output
      inputProcess->setSrc(color, albedo, normal);
      outputProcess->setDst(outputTemp ? outputTemp : output);

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
      {
        imageCopy->setDst(output);
        imageCopy->run();
      }

      // Finished
      progress.finish();
    });

    if (sync)
      device->wait();
  }

  void UNetFilter::init()
  {
    cleanup();
    checkParams();

    // Build the network
    net.reset(new Network(device, parseWeights()));
    transferFunc = newTransferFunc();
    
    // Divide the image into tiles until the memory usage gets below the threshold
    const int minTileSize = 3*overlap;
    const size_t maxScratchByteSize = size_t(maxMemoryMB)*1024*1024;

    H = output->getH();
    W = output->getW();
    tileCountH = 1;
    tileCountW = 1;
    tileH = round_up(H, alignment);
    tileW = round_up(W, alignment);

    while (!buildNet(maxScratchByteSize))
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
      {
        // Cannot divide further
        if (!buildNet())
          throw std::runtime_error("could not build network");
        break;
      }
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

  void UNetFilter::cleanup()
  {
    net = nullptr;
    autoexposure.reset();
    inputProcess.reset();
    outputProcess.reset();
    imageCopy.reset();
    transferFunc.reset();
    outputTemp.reset();
  }

  void UNetFilter::checkParams()
  {
    if (!color && !albedo && !normal)
      throw Exception(Error::InvalidOperation, "input image not specified");
    if (!output)
      throw Exception(Error::InvalidOperation, "output image not specified");

    if (((color  && color->getFormat()  != Format::Float3) ||
         (albedo && albedo->getFormat() != Format::Float3) ||
         (normal && normal->getFormat() != Format::Float3)) &&
        ((color  && color->getFormat()  != Format::Half3) ||
         (albedo && albedo->getFormat() != Format::Half3) ||
         (normal && normal->getFormat() != Format::Half3)))
      throw Exception(Error::InvalidOperation, "unsupported input image format");

    if (output->getFormat() != Format::Float3 && output->getFormat() != Format::Half3)
      throw Exception(Error::InvalidOperation, "unsupported output image format");

    if ((color  && (color->getW()  != output->getW() || color->getH()  != output->getH())) ||
        (albedo && (albedo->getW() != output->getW() || albedo->getH() != output->getH())) ||
        (normal && (normal->getW() != output->getW() || normal->getH() != output->getH())))
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
  }

  std::shared_ptr<Weights> UNetFilter::parseWeights()
  {
    // Select the weights to parse
    Data weightsBlob;

    if (color)
    {
      if (!albedo && !normal)
      {
        weightsBlob = directional ? weightsBlobs.dir : (hdr ? weightsBlobs.hdr : weightsBlobs.ldr);
      }
      else if (albedo && !normal)
      {
        weightsBlob = hdr ? weightsBlobs.hdr_alb : weightsBlobs.ldr_alb;
      }
      else if (albedo && normal)
      {
        if (cleanAux)
          weightsBlob = hdr ? weightsBlobs.hdr_calb_cnrm : weightsBlobs.ldr_calb_cnrm;
        else
          weightsBlob = hdr ? weightsBlobs.hdr_alb_nrm : weightsBlobs.ldr_alb_nrm;
      }
    }
    else
    {
      // Auxiliary feature filtering
      if (albedo && !normal)
      {
        if (hdr)
          throw Exception(Error::InvalidOperation, "hdr mode is not supported for albedo filtering");
        weightsBlob = weightsBlobs.alb;
      }
      else if (!albedo && normal)
      {
        if (hdr || srgb)
          throw Exception(Error::InvalidOperation, "hdr and srgb modes are not supported for normal filtering");
        weightsBlob = weightsBlobs.nrm;
      }
      else
      {
        throw Exception(Error::InvalidOperation, "invalid combination of input features");
      }
    }

    if (userWeightsBlob)
      weightsBlob = userWeightsBlob;

    if (!weightsBlob)
      throw Exception(Error::InvalidOperation, "unsupported combination of input features");

    return std::make_shared<Weights>(device, weightsBlob);
  }

  // Tries to build the network without exceeding the specified amount of scratch memory
  bool UNetFilter::buildNet(size_t maxScratchByteSize)
  {
    // If the image size is zero, there is nothing else to do
    if (H <= 0 || W <= 0)
      return true;

    // Get the number of input channels
    int inputC = 0;
    if (color)  inputC += color->getC();
    if (albedo) inputC += albedo->getC();
    if (normal) inputC += normal->getC();

    // Create the ops
    std::shared_ptr<Autoexposure> autoexposure;
    if (hdr)
      autoexposure = device->newAutoexposure(color->getDesc());

    const bool snorm = directional || (!color && normal);
    TensorDims inputDims {inputC, tileH, tileW};
    auto inputProcess = net->addInputProcess("input", inputDims, alignment, transferFunc, hdr, snorm);

    auto encConv0 = net->addConv("enc_conv0", inputProcess->getDstDesc());

    auto encConv1 = net->addConv("enc_conv1", encConv0->getDstDesc());
    auto pool1    = net->addPool("pool1", encConv1->getDstDesc());

    auto encConv2 = net->addConv("enc_conv2", pool1->getDstDesc());
    auto pool2    = net->addPool("pool2", encConv2->getDstDesc());

    auto encConv3 = net->addConv("enc_conv3", pool2->getDstDesc());
    auto pool3    = net->addPool("pool3", encConv3->getDstDesc());

    auto encConv4 = net->addConv("enc_conv4", pool3->getDstDesc());
    auto pool4    = net->addPool("pool4", encConv4->getDstDesc());

    auto encConv5a = net->addConv("enc_conv5a", pool4->getDstDesc());
    auto encConv5b = net->addConv("enc_conv5b", encConv5a->getDstDesc());

    auto upsample4 = net->addUpsample("upsample4", encConv5b->getDstDesc());
    auto decConv4a = net->addConcatConv("dec_conv4a", upsample4->getDstDesc(), pool3->getDstDesc());
    auto decConv4b = net->addConv("dec_conv4b", decConv4a->getDstDesc());

    auto upsample3 = net->addUpsample("upsample3", decConv4b->getDstDesc());
    auto decConv3a = net->addConcatConv("dec_conv3a", upsample3->getDstDesc(), pool2->getDstDesc());
    auto decConv3b = net->addConv("dec_conv3b", decConv3a->getDstDesc());

    auto upsample2 = net->addUpsample("upsample2", decConv3b->getDstDesc());
    auto decConv2a = net->addConcatConv("dec_conv2a", upsample2->getDstDesc(), pool1->getDstDesc());
    auto decConv2b = net->addConv("dec_conv2b", decConv2a->getDstDesc());

    auto upsample1 = net->addUpsample("upsample1", decConv2b->getDstDesc());
    auto decConv1a = net->addConcatConv("dec_conv1a", upsample1->getDstDesc(), inputProcess->getDstDesc());
    auto decConv1b = net->addConv("dec_conv1b", decConv1a->getDstDesc());
    
    auto decConv0 = net->addConv("dec_conv0", decConv1b->getDstDesc(), Activation::None);

    auto outputProcess = net->addOutputProcess("output", decConv0->getDstDesc(), transferFunc, hdr, snorm);

    if (!net->isSupported())
    {
      net->clear();
      return false;
    }

    // Compute the tensor offsets
    ptrdiff_t endOfs = 0; // we'll have negative offsets relative to the end of the buffer
    ptrdiff_t inputOfs     = endOfs - inputProcess->getDstDesc().getAlignedSize();
    ptrdiff_t encConv0Ofs  = inputOfs - encConv0->getDstDesc().getAlignedSize();
    ptrdiff_t pool1Ofs     = inputOfs - pool1->getDstDesc().getAlignedSize();
    ptrdiff_t encConv1Ofs  = min(encConv0Ofs, pool1Ofs) - encConv1->getDstDesc().getAlignedSize();
    ptrdiff_t pool2Ofs     = pool1Ofs - pool2->getDstDesc().getAlignedSize();
    ptrdiff_t encConv2Ofs  = pool2Ofs - encConv2->getDstDesc().getAlignedSize();
    ptrdiff_t pool3Ofs     = pool2Ofs - pool3->getDstDesc().getAlignedSize();
    ptrdiff_t encConv3Ofs  = pool3Ofs - encConv3->getDstDesc().getAlignedSize();
    ptrdiff_t encConv4Ofs  = pool3Ofs - encConv4->getDstDesc().getAlignedSize();
    ptrdiff_t encConv5aOfs = pool3Ofs - encConv5a->getDstDesc().getAlignedSize();
    ptrdiff_t pool4Ofs     = min(encConv4Ofs, encConv5aOfs) - pool4->getDstDesc().getAlignedSize();
    ptrdiff_t upsample4Ofs = pool3Ofs - upsample4->getDstDesc().getAlignedSize();
    ptrdiff_t encConv5bOfs = min(encConv5aOfs, upsample4Ofs) - encConv5b->getDstDesc().getAlignedSize();
    ptrdiff_t upsample3Ofs = pool2Ofs - upsample3->getDstDesc().getAlignedSize();
    ptrdiff_t decConv4bOfs = upsample3Ofs - decConv4b->getDstDesc().getAlignedSize();
    ptrdiff_t decConv4aOfs = min(upsample4Ofs, decConv4bOfs) - decConv4a->getDstDesc().getAlignedSize();
    ptrdiff_t upsample2Ofs = pool1Ofs - upsample2->getDstDesc().getAlignedSize();
    ptrdiff_t decConv3bOfs = upsample2Ofs - decConv3b->getDstDesc().getAlignedSize();
    ptrdiff_t decConv3aOfs = min(upsample3Ofs, decConv3bOfs) - decConv3a->getDstDesc().getAlignedSize();
    ptrdiff_t upsample1Ofs = inputOfs - upsample1->getDstDesc().getAlignedSize();
    ptrdiff_t decConv2bOfs = upsample1Ofs - decConv2b->getDstDesc().getAlignedSize();
    ptrdiff_t decConv2aOfs = min(upsample2Ofs, decConv2bOfs) - decConv2a->getDstDesc().getAlignedSize();
    ptrdiff_t decConv1bOfs = endOfs - decConv1b->getDstDesc().getAlignedSize();
    ptrdiff_t decConv1aOfs = min(upsample1Ofs, decConv1bOfs) - decConv1a->getDstDesc().getAlignedSize();
    ptrdiff_t decConv0Ofs  = decConv1bOfs - decConv0->getDstDesc().getAlignedSize();

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

    // If doing in-place _tiled_ filtering, allocate a temporary output image
    ImageDesc outputTempDesc(output->getFormat(), W, H);
    ptrdiff_t outputTempOfs = 0;
    if (inplace && (tileCountH * tileCountW) > 1)
    {
      outputTempOfs = minOfs - outputTempDesc.getAlignedSize();
      minOfs = outputTempOfs;
    }

    // Allocate scratch space for the operations
    size_t opScratchByteSize = net->getScratchAlignedSize();
    if (hdr)
      opScratchByteSize = max(opScratchByteSize, autoexposure->getScratchAlignedSize());
    ptrdiff_t opScratchOfs = minOfs - opScratchByteSize;
    minOfs = opScratchOfs;

    // Compute the size of the scratch buffer
    const size_t scratchByteSize = -minOfs;

    // Check the total memory usage
    const size_t weightsByteSize = net->getWeights()->getScratchByteSize();
    const size_t totalScratchByteSize = scratchByteSize + weightsByteSize;
    if (totalScratchByteSize > maxScratchByteSize)
    {
      net->clear();
      return false;
    }

    // Allocate the scratch buffer
    auto scratch = device->newScratchBuffer(scratchByteSize);

    // Connect the network layers
    inputProcess->setDst(scratch->newTensor(inputProcess->getDstDesc(), inputOfs));

    encConv0->setSrc(inputProcess->getDst());
    encConv0->setDst(scratch->newTensor(encConv0->getDstDesc(), encConv0Ofs));

    encConv1->setSrc(encConv0->getDst());
    encConv1->setDst(scratch->newTensor(encConv1->getDstDesc(), encConv1Ofs));

    pool1->setSrc(encConv1->getDst());
    pool1->setDst(scratch->newTensor(pool1->getDstDesc(), pool1Ofs));

    encConv2->setSrc(pool1->getDst());
    encConv2->setDst(scratch->newTensor(encConv2->getDstDesc(), encConv2Ofs));

    pool2->setSrc(encConv2->getDst());
    pool2->setDst(scratch->newTensor(pool2->getDstDesc(), pool2Ofs));

    encConv3->setSrc(pool2->getDst());
    encConv3->setDst(scratch->newTensor(encConv3->getDstDesc(), encConv3Ofs));

    pool3->setSrc(encConv3->getDst());
    pool3->setDst(scratch->newTensor(pool3->getDstDesc(), pool3Ofs));

    encConv4->setSrc(pool3->getDst());
    encConv4->setDst(scratch->newTensor(encConv4->getDstDesc(), encConv4Ofs));

    pool4->setSrc(encConv4->getDst());
    pool4->setDst(scratch->newTensor(pool4->getDstDesc(), pool4Ofs));

    encConv5a->setSrc(pool4->getDst());
    encConv5a->setDst(scratch->newTensor(encConv5a->getDstDesc(), encConv5aOfs));

    encConv5b->setSrc(encConv5a->getDst());
    encConv5b->setDst(scratch->newTensor(encConv5b->getDstDesc(), encConv5bOfs));

    upsample4->setSrc(encConv5b->getDst());
    upsample4->setDst(scratch->newTensor(upsample4->getDstDesc(), upsample4Ofs));

    decConv4a->setSrc(upsample4->getDst(), pool3->getDst());
    decConv4a->setDst(scratch->newTensor(decConv4a->getDstDesc(), decConv4aOfs));

    decConv4b->setSrc(decConv4a->getDst());
    decConv4b->setDst(scratch->newTensor(decConv4b->getDstDesc(), decConv4bOfs));

    upsample3->setSrc(decConv4b->getDst());
    upsample3->setDst(scratch->newTensor(upsample3->getDstDesc(), upsample3Ofs));

    decConv3a->setSrc(upsample3->getDst(), pool2->getDst());
    decConv3a->setDst(scratch->newTensor(decConv3a->getDstDesc(), decConv3aOfs));

    decConv3b->setSrc(decConv3a->getDst());
    decConv3b->setDst(scratch->newTensor(decConv3b->getDstDesc(), decConv3bOfs));

    upsample2->setSrc(decConv3b->getDst());
    upsample2->setDst(scratch->newTensor(upsample2->getDstDesc(), upsample2Ofs));

    decConv2a->setSrc(upsample2->getDst(), pool1->getDst());
    decConv2a->setDst(scratch->newTensor(decConv2a->getDstDesc(), decConv2aOfs));

    decConv2b->setSrc(decConv2a->getDst());
    decConv2b->setDst(scratch->newTensor(decConv2b->getDstDesc(), decConv2bOfs));

    upsample1->setSrc(decConv2b->getDst());
    upsample1->setDst(scratch->newTensor(upsample1->getDstDesc(), upsample1Ofs));

    decConv1a->setSrc(upsample1->getDst(), inputProcess->getDst());
    decConv1a->setDst(scratch->newTensor(decConv1a->getDstDesc(), decConv1aOfs));

    decConv1b->setSrc(decConv1a->getDst());
    decConv1b->setDst(scratch->newTensor(decConv1b->getDstDesc(), decConv1bOfs));

    decConv0->setSrc(decConv1b->getDst());
    decConv0->setDst(scratch->newTensor(decConv0->getDstDesc(), decConv0Ofs));

    outputProcess->setSrc(decConv0->getDst());

    // Create the temporary output
    if (outputTempOfs)
      outputTemp = scratch->newImage(outputTempDesc, outputTempOfs);

    // Set the scratch buffer for the operations
    if (opScratchByteSize > 0)
    {
      TensorDesc opScratchDesc {{int64_t(opScratchByteSize)}, TensorLayout::x, DataType::UInt8};
      auto opScratch = scratch->newTensor(opScratchDesc, opScratchOfs);
      net->setScratch(opScratch);
      if (hdr)
        autoexposure->setScratch(opScratch);
    }

    // Finalize the network
    net->finalize();
    if (hdr)
      autoexposure->finalize();
    this->autoexposure = autoexposure;
    this->inputProcess = inputProcess;
    this->outputProcess = outputProcess;
    if (outputTemp)
    {
      imageCopy = device->newImageCopy();
      imageCopy->setSrc(outputTemp);
      imageCopy->finalize();
    }

    // Print statistics
    if (device->isVerbose(2))
    {
      std::cout << "Weight scratch bytes: " << weightsByteSize << std::endl;
      std::cout << "Tensor scratch bytes: " << (scratchByteSize - opScratchByteSize) << std::endl;
      std::cout << "Op scratch bytes    : " << opScratchByteSize << std::endl;
      std::cout << "Total scratch bytes : " << totalScratchByteSize << std::endl;
    }

    return true;
  }

} // namespace oidn
