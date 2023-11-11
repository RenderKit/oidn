// Copyright 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "filter.h"
#include "graph.h"
#include "color.h"
#include "image_copy.h"

OIDN_NAMESPACE_BEGIN

  // U-Net based denoising filter
  class UNetFilter : public Filter
  {
  public:
    void setData(const std::string& name, const Data& data) override;
    void updateData(const std::string& name) override;
    void unsetData(const std::string& name) override;
    void setInt(const std::string& name, int value) override;
    int getInt(const std::string& name) override;
    void setFloat(const std::string& name, float value) override;
    float getFloat(const std::string& name) override;

    void commit() override;
    void execute(SyncMode sync) override;

  protected:
    explicit UNetFilter(const Ref<Device>& device);
    virtual std::shared_ptr<TransferFunction> newTransferFunc() = 0;

    // Network constants
    // TODO: autodetect these values from the model
    static constexpr int receptiveField  = 174; // receptive field in pixels
    static constexpr int tileAlignment   = 16;  // required spatial alignment in pixels (padding may be necessary)
    static constexpr int tileOverlap     = round_up(receptiveField / 2, tileAlignment); // required spatial overlap between tiles in pixels
    static constexpr int defaultMaxTileSize = 2160*2160; // default maximum number of pixels per tile

    // Images
    Ref<Image> color;
    Ref<Image> albedo;
    Ref<Image> normal;
    Ref<Image> output;

    // Options
    static constexpr Quality defaultQuality = Quality::High;
    Quality quality = defaultQuality;
    bool hdr = false;
    bool srgb = false;
    bool directional = false;
    float inputScale = std::numeric_limits<float>::quiet_NaN();
    bool cleanAux = false;
    int maxMemoryMB = -1; // maximum memory usage limit in MBs, disabled if < 0

    // Weights
    struct
    {
      Data hdr;
      Data hdr_alb;
      Data hdr_alb_nrm;
      Data hdr_calb_cnrm;
      Data ldr;
      Data ldr_alb;
      Data ldr_alb_nrm;
      Data ldr_calb_cnrm;
      Data dir;
      Data alb;
      Data nrm;
    } weightsBlobs;
    Data userWeightsBlob;

  private:
    void init();
    void cleanup();
    void checkParams();
    Data getWeights();
    bool buildModel(size_t maxMemoryByteSize = std::numeric_limits<size_t>::max());
    void resetModel();

    // Image dimensions
    int H = 0;            // image height
    int W = 0;            // image width
    int tileH = 0;        // tile height
    int tileW = 0;        // tile width
    int tileCountH = 1;   // number of tiles in H dimension
    int tileCountW = 1;   // number of tiles in W dimension
    bool inplace = false; // indicates whether input and output buffers overlap

    // Per-engine model instance
    struct Instance
    {
      std::shared_ptr<Graph> graph;
      std::shared_ptr<InputProcess> inputProcess;
      std::shared_ptr<OutputProcess> outputProcess;
    };

    // Model
    std::vector<Instance> instances;
    std::shared_ptr<TransferFunction> transferFunc;
    std::shared_ptr<Autoexposure> autoexposure;
    // In-place tiled filtering
    std::shared_ptr<ImageCopy> imageCopy;
    Ref<Image> outputTemp;

    Progress progress;
  };

OIDN_NAMESPACE_END
