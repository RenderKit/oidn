// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "unet_filter.h"
#include "tza.h"

OIDN_NAMESPACE_BEGIN

  UNetFilter::UNetFilter(const Ref<Device>& device)
    : Filter(device)
  {
    // Compute final device-dependent tile alignment and overlap
    tileAlignment = lcm(minTileAlignment, device->getMinTileAlignment());
    tileOverlap = round_up(receptiveField / 2, tileAlignment);
  }

  void UNetFilter::setData(const std::string& name, const Data& data)
  {
    if (name == "weights")
      setParam(userWeightsBlob, data);
    else
      device->printWarning("unknown filter parameter or type mismatch: '" + name + "'");

    dirty = true;
  }

  void UNetFilter::updateData(const std::string& name)
  {
    if (name == "weights")
      dirtyParam |= userWeightsBlob;
    else
      device->printWarning("unknown filter parameter or type mismatch: '" + name + "'");

    dirty = true;
  }

  void UNetFilter::unsetData(const std::string& name)
  {
    if (name == "weights")
      removeParam(userWeightsBlob);
    else
      device->printWarning("unknown filter parameter or type mismatch: '" + name + "'");

    dirty = true;
  }

  void UNetFilter::setInt(const std::string& name, int value)
  {
    if (name == "quality")
    {
      Quality qualityValue = static_cast<Quality>(value);
      if (qualityValue == Quality::Default)
        qualityValue = defaultQuality;
      else if (qualityValue != Quality::High && qualityValue != Quality::Balanced)
        throw Exception(Error::InvalidArgument, "unknown filter quality mode");
      setParam(quality, qualityValue);
    }
    else if (name == "maxMemoryMB")
      setParam(maxMemoryMB, value);
    else
      device->printWarning("unknown filter parameter or type mismatch: '" + name + "'");

    dirty = true;
  }

  int UNetFilter::getInt(const std::string& name)
  {
    if (name == "quality")
      return static_cast<int>(quality);
    else if (name == "maxMemoryMB")
      return maxMemoryMB;
    else if (name == "tileAlignment")
      return tileAlignment;
    else if (name == "alignment")
    {
      device->printWarning("filter parameter 'alignment' is deprecated, use 'tileAlignment' instead");
      return tileAlignment;
    }
    else if (name == "tileOverlap")
      return tileOverlap;
    else if (name == "overlap")
    {
      device->printWarning("filter parameter 'overlap' is deprecated, use 'tileOverlap' instead");
      return tileOverlap;
    }
    else
      throw Exception(Error::InvalidArgument, "unknown filter parameter or type mismatch: '" + name + "'");
  }

  void UNetFilter::setFloat(const std::string& name, float value)
  {
    if (name == "inputScale")
      inputScale = value;
    else if (name == "hdrScale")
    {
      device->printWarning("filter parameter 'hdrScale' is deprecated, use 'inputScale' instead");
      inputScale = value;
    }
    else
      device->printWarning("unknown filter parameter or type mismatch: '" + name + "'");

    dirty = true;
  }

  float UNetFilter::getFloat(const std::string& name)
  {
    if (name == "inputScale")
      return inputScale;
    else if (name == "hdrScale")
    {
      device->printWarning("filter parameter 'hdrScale' is deprecated, use 'inputScale' instead");
      return inputScale;
    }
    else
      throw Exception(Error::InvalidArgument, "unknown filter parameter or type mismatch: '" + name + "'");
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
      device->getEngine()->runHostTask([&]() { init(); });
      device->wait();

      // Clean up the device memory if the memory usage limit has been reduced
      if (maxMemoryMB >= 0 && (maxMemoryMB < prevMaxMemoryMB || prevMaxMemoryMB < 0))
        device->trimScratch();
      prevMaxMemoryMB = maxMemoryMB;
    }

    dirty = false;
    dirtyParam = false;
  }

  void UNetFilter::execute(SyncMode sync)
  {
    if (dirty)
      throw Exception(Error::InvalidOperation, "changes to the filter are not committed");

    if (H <= 0 || W <= 0)
      return;

    auto mainEngine = device->getEngine();

    mainEngine->runHostTask([&]()
    {
      // Initialize the progress state
      double workAmount = tileCountH * tileCountW * instances[0].graph->getWorkAmount();
      if (hdr && math::isnan(inputScale))
        workAmount += 1;
      if (outputTemp)
        workAmount += 1;
      progress.start(mainEngine, progressFunc, progressUserPtr, workAmount);

      // Set the input scale
      if (math::isnan(inputScale))
      {
        if (hdr)
        {
          autoexposure->setSrc(color);
          autoexposure->submit();
          device->submitBarrier();
          progress.update(mainEngine, 1);
          transferFunc->setInputScale(autoexposure->getDstPtr());
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
      for (auto& instance : instances)
      {
        instance.inputProcess->setSrc(color, albedo, normal);
        instance.outputProcess->setDst(outputTemp ? outputTemp : output);
      }

      // Iterate over the tiles
      int tileIndex = 0;

      for (int i = 0; i < tileCountH; ++i)
      {
        const int h = i * (tileH - (2*tileOverlap+tilePadH)); // input tile position (including overlaps)
        const int overlapBeginH = i > 0            ? tileOverlap : 0; // overlap on the top
        const int overlapEndH   = i < tileCountH-1 ? tileOverlap+tilePadH : 0; // overlap on the bottom
        const int tileH1 = min(H - h, tileH); // input tile size (including overlaps)
        const int tileH2 = tileH1 - overlapBeginH - overlapEndH; // output tile size
        const int alignOffsetH = tileH - round_up(tileH1, minTileAlignment); // align to the bottom in the tile buffer

        for (int j = 0; j < tileCountW; ++j)
        {
          const int w = j * (tileW - (2*tileOverlap+tilePadW)); // input tile position (including overlaps)
          const int overlapBeginW = j > 0            ? tileOverlap : 0; // overlap on the left
          const int overlapEndW   = j < tileCountW-1 ? tileOverlap+tilePadW : 0; // overlap on the right
          const int tileW1 = min(W - w, tileW); // input tile size (including overlaps)
          const int tileW2 = tileW1 - overlapBeginW - overlapEndW; // output tile size
          const int alignOffsetW = tileW - round_up(tileW1, minTileAlignment); // align to the right in the tile buffer

          auto& instance = instances[tileIndex % device->getNumEngines()];

          // Set the input tile
          instance.inputProcess->setTile(
            h, w,
            alignOffsetH, alignOffsetW,
            tileH1, tileW1);

          // Set the output tile
          instance.outputProcess->setTile(
            alignOffsetH + overlapBeginH, alignOffsetW + overlapBeginW,
            h + overlapBeginH, w + overlapBeginW,
            tileH2, tileW2);

          //printf("Tile: %d %d -> %d %d\n", w+overlapBeginW, h+overlapBeginH, w+overlapBeginW+tileW2, h+overlapBeginH+tileH2);

          // Denoise the tile
          instance.graph->run(progress);

          // Next tile
          tileIndex++;
        }
      }

      device->submitBarrier();

      // Copy the output image to the final buffer if filtering in-place
      if (outputTemp)
      {
        imageCopy->setDst(output);
        imageCopy->submit();
      }

      // Finished
      progress.finish(mainEngine);
    });

    if (sync == SyncMode::Sync)
      device->wait();
    else
      device->flush();
  }

  void UNetFilter::init()
  {
    cleanup();
    checkParams();

    // Build the model
    Data weightsBlob = getWeights();
    auto constTensors = parseTZA(weightsBlob.ptr, weightsBlob.size);
    const bool fastMath = quality == Quality::Balanced;

    for (int i = 0; i < device->getNumEngines(); ++i)
    {
      Engine* engine = device->getEngine(i);

      // We can use cached weights only for built-in weights because user weights may change!
      auto cachedConstTensors = userWeightsBlob ? nullptr : engine->getCachedTensors(weightsBlob.ptr);
      
      instances.emplace_back();
      instances.back().graph = makeRef<Graph>(engine, constTensors, cachedConstTensors, fastMath);
    }

    transferFunc = newTransferFunc();

    // Try to divide the image into tiles until the memory usage gets below the specified threshold
    // and the number of tiles is a multiple of the number of engines
    H = output->getH();
    W = output->getW();
    tileH = round_up(H, minTileAlignment); // add minimum device-independent padding
    tileW = round_up(W, minTileAlignment);
    tilePadH = tileH % tileAlignment; // increase the overlap on the bottom to align offsets
    tilePadW = tileW % tileAlignment; // increase the overlap on the right to align offsets
    tileCountH = 1;
    tileCountW = 1;

    const int minTileDim = max(4*tileOverlap, 768); // MPS has slightly different output using smaller tiles
    const int minTileH = round_up(minTileDim, tileAlignment, tilePadH);
    const int minTileW = round_up(minTileDim, tileAlignment, tilePadW);

    const int maxTileSize = (maxMemoryMB < 0) ? defaultMaxTileSize : INT_MAX;
    const size_t maxMemoryByteSize = (maxMemoryMB >= 0) ? size_t(maxMemoryMB)*1024*1024 : SIZE_MAX;

    while ((tileCountH * tileCountW) % device->getNumEngines() != 0 ||
           (tileH * tileW) > maxTileSize ||
           !buildModel(maxMemoryByteSize))
    {
      if (tileH > minTileH && tileH > tileW)
      {
        const int newTileH = ceil_div(H + (2*tileOverlap+tilePadH) * tileCountH, tileCountH + 1);
        tileH = clamp(round_up(newTileH, tileAlignment, tilePadH), minTileH, tileH - tileAlignment);
        tileCountH = max(ceil_div(H - (2*tileOverlap+tilePadH), tileH - (2*tileOverlap+tilePadH)), 1);
      }
      else if (tileW > minTileW)
      {
        const int newTileW = ceil_div(W + (2*tileOverlap+tilePadW) * tileCountW, tileCountW + 1);
        tileW = clamp(round_up(newTileW, tileAlignment, tilePadW), minTileW, tileW - tileAlignment);
        tileCountW = max(ceil_div(W - (2*tileOverlap+tilePadW), tileW - (2*tileOverlap+tilePadW)), 1);
      }
      else
      {
        // Cannot divide further
        if (!buildModel())
          throw std::runtime_error("could not build filter model");
        break;
      }
    }

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
    instances.clear();
    transferFunc.reset();
    autoexposure.reset();
    imageCopy.reset();
    outputTemp.reset();
  }

  void UNetFilter::checkParams()
  {
    if (!color && !albedo && !normal)
      throw Exception(Error::InvalidOperation, "input image not specified");
    if (!output)
      throw Exception(Error::InvalidOperation, "output image not specified");

    auto isSupportedFormat = [](Format format)
    {
      return format == Format::Float3 || format == Format::Half3 ||
             format == Format::Float2 || format == Format::Half2 ||
             format == Format::Float  || format == Format::Half;
    };

    if ((color  && !isSupportedFormat(color->getFormat()))  ||
        (albedo && !isSupportedFormat(albedo->getFormat())) ||
        (normal && !isSupportedFormat(normal->getFormat())))
      throw Exception(Error::InvalidOperation, "unsupported input image format");

    if (!isSupportedFormat(output->getFormat()))
      throw Exception(Error::InvalidOperation, "unsupported output image format");

    Image* input = color ? color.get() : (albedo ? albedo.get() : normal.get());
    if (input->getC() != output->getC())
      throw Exception(Error::InvalidOperation, "input/output image channel count mismatch");

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
      std::cout << "Quality: " << quality << std::endl;
      std::cout << "Inputs:";
      if (color)  std::cout << " " << (directional ? "dir" : (hdr ? "hdr" : "ldr")) << ":" << color->getFormat();
      if (albedo) std::cout << " " << "alb" << ":" << albedo->getFormat();
      if (normal) std::cout << " " << "nrm" << ":" << normal->getFormat();
      std::cout << std::endl;
      std::cout << "Output: " << output->getFormat() << std::endl;
    }
  }

  Data UNetFilter::getWeights()
  {
    // Select the weights to use
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

    return weightsBlob;
  }

  // Tries to build the model without exceeding the specified amount of memory
  bool UNetFilter::buildModel(size_t maxMemoryByteSize)
  {
    // If the image size is zero, there is nothing else to do
    if (H <= 0 || W <= 0)
      return true;

    // Get the number of input channels
    int inputC = 0;
    if (color)  inputC += 3; // always broadcast to 3 channels
    if (albedo) inputC += 3;
    if (normal) inputC += 3;

    // Create global operations (not part of any model instance or graph)
    Ref<Autoexposure> autoexposure;
    if (hdr)
      autoexposure = device->getEngine()->newAutoexposure(color->getDesc());

    const bool snorm = directional || (!color && normal);
    TensorDims inputDims{inputC, tileH, tileW};
    size_t totalMemoryByteSize = 0;

    // Create model instances for each engine of the device
    for (int instanceID = 0; instanceID < device->getNumEngines(); ++instanceID)
    {
      auto& instance = instances[instanceID];
      auto& graph = instance.graph;

      // Create the model graph
      auto inputProcess = graph->addInputProcess("input", inputDims,
                                                 transferFunc, hdr, snorm);

      auto encConv0 = graph->addConv("enc_conv0", inputProcess, Activation::ReLU);

      auto pool1 = graph->addConv("enc_conv1", encConv0, Activation::ReLU, PostOp::Pool);

      auto pool2 = graph->addConv("enc_conv2", pool1, Activation::ReLU, PostOp::Pool);

      auto pool3 = graph->addConv("enc_conv3", pool2, Activation::ReLU, PostOp::Pool);

      auto pool4 = graph->addConv("enc_conv4", pool3, Activation::ReLU, PostOp::Pool);

      auto encConv5a = graph->addConv("enc_conv5a", pool4, Activation::ReLU);

      auto upsample4 = graph->addConv("enc_conv5b", encConv5a, Activation::ReLU, PostOp::Upsample);
      auto decConv4a = graph->addConcatConv("dec_conv4a", upsample4, pool3, Activation::ReLU);

      auto upsample3 = graph->addConv("dec_conv4b", decConv4a, Activation::ReLU, PostOp::Upsample);
      auto decConv3a = graph->addConcatConv("dec_conv3a", upsample3, pool2, Activation::ReLU);

      auto upsample2 = graph->addConv("dec_conv3b", decConv3a, Activation::ReLU, PostOp::Upsample);
      auto decConv2a = graph->addConcatConv("dec_conv2a", upsample2, pool1, Activation::ReLU);

      auto upsample1 = graph->addConv("dec_conv2b", decConv2a, Activation::ReLU, PostOp::Upsample);
      auto decConv1a = graph->addConcatConv("dec_conv1a", upsample1, inputProcess, Activation::ReLU);
      auto decConv1b = graph->addConv("dec_conv1b", decConv1a, Activation::ReLU);

      auto decConv0 = graph->addConv("dec_conv0", decConv1b, Activation::ReLU);

      auto outputProcess = graph->addOutputProcess("output", decConv0, transferFunc, hdr, snorm);

      // Check whether all operations in the graph are supported
      if (!graph->isSupported())
      {
        resetModel();
        return false;
      }

      // Get the scratch size of the graph
      const size_t graphScratchByteSize = round_up(graph->getScratchByteSize(), memoryAlignment);
      size_t scratchByteSize = graphScratchByteSize;

      // Allocate scratch for global operations
      if (instanceID == 0 && hdr)
        scratchByteSize = max(scratchByteSize, autoexposure->getScratchByteSize());

      scratchByteSize = round_up(scratchByteSize, memoryAlignment);

      // If doing in-place _tiled_ filtering, allocate a temporary output image
      ImageDesc outputTempDesc(output->getFormat(), W, H);
      size_t outputTempByteOffset = SIZE_MAX;
      if (instanceID == 0 && inplace && (tileCountH * tileCountW) > 1)
      {
        outputTempByteOffset = scratchByteSize;
        scratchByteSize += round_up(outputTempDesc.getByteSize(), memoryAlignment);
      }

      // If denoising in HDR mode, allocate a tensor for the autoexposure result
      size_t autoexposureDstOffset = SIZE_MAX;
      if (instanceID == 0 && hdr)
      {
        autoexposureDstOffset = scratchByteSize;
        scratchByteSize += round_up(sizeof(float), memoryAlignment);
      }

      // Check the total memory usage
      if (instanceID == 0)
      {
        totalMemoryByteSize = (scratchByteSize + graph->getPrivateByteSize()) +
          (graphScratchByteSize + graph->getPrivateByteSize()) * (device->getNumEngines() - 1);

        if (totalMemoryByteSize > maxMemoryByteSize)
        {
          resetModel();
          return false;
        }
      }

      // Allocate the scratch buffer
      auto scratchArena = device->getEngine(instanceID)->newScratchArena(scratchByteSize);
      auto scratch = scratchArena->newBuffer(scratchByteSize);

      // Set the scratch buffer for the graph and the global operations
      graph->setScratch(scratch);
      if (instanceID == 0 && hdr)
      {
        autoexposure->setScratch(scratch);
        autoexposure->setDst(makeRef<Record<float>>(scratch, autoexposureDstOffset));
      }

      // Finalize the network
      graph->finalize();

      // Create the temporary output image
      if (instanceID == 0 && outputTempByteOffset < SIZE_MAX)
        outputTemp = scratch->newImage(outputTempDesc, outputTempByteOffset);

      instance.inputProcess  = inputProcess;
      instance.outputProcess = outputProcess;
    }

    // Finalize the global operations
    if (hdr)
      autoexposure->finalize();
    this->autoexposure = autoexposure;

    if (outputTemp)
    {
      imageCopy = device->getEngine()->newImageCopy();
      imageCopy->setSrc(outputTemp);
      imageCopy->finalize();
    }

    // Print statistics
    if (device->isVerbose(2))
      std::cout << "Memory usage: " << totalMemoryByteSize << std::endl;

    return true;
  }

  void UNetFilter::resetModel()
  {
    for (auto& instance : instances)
    {
      instance.graph->clear();
      instance.inputProcess.reset();
      instance.outputProcess.reset();
    }

    autoexposure.reset();
    imageCopy.reset();
    outputTemp.reset();
  }

OIDN_NAMESPACE_END
