// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "filter.h"
#include "network.h"
#include "scratch.h"
#include "color.h"

namespace oidn {

  // U-Net based denoising filter
  class UNetFilter : public Filter
  {
  public:
    void setData(const std::string& name, const Data& data) override;
    void updateData(const std::string& name) override;
    void removeData(const std::string& name) override;
    void set1f(const std::string& name, float value) override;
    float get1f(const std::string& name) override;

    void commit() override;
    void execute(bool sync) override;

  protected:
    explicit UNetFilter(const Ref<Device>& device);
    virtual std::shared_ptr<TransferFunction> newTransferFunc() = 0;

    // Network constants
    static constexpr int alignment       = 16;  // required spatial alignment in pixels (padding may be necessary)
    static constexpr int receptiveField  = 174; // receptive field in pixels
    static constexpr int overlap         = round_up(receptiveField / 2, alignment); // required spatial overlap between tiles in pixels

    // Images
    std::shared_ptr<Image> color;
    std::shared_ptr<Image> albedo;
    std::shared_ptr<Image> normal;
    std::shared_ptr<Image> output;
    std::shared_ptr<Image> outputTemp; // required for in-place tiled filtering

    // Options
    bool hdr = false;
    bool srgb = false;
    bool directional = false;
    float inputScale = std::numeric_limits<float>::quiet_NaN();
    bool cleanAux = false;
    int maxMemoryMB = 3000; // approximate maximum memory usage in MBs

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
    std::shared_ptr<Weights> parseWeights();
    bool buildNet(size_t maxScratchByteSize = std::numeric_limits<size_t>::max());

    // Image dimensions
    int H = 0;            // image height
    int W = 0;            // image width
    int tileH = 0;        // tile height
    int tileW = 0;        // tile width
    int tileCountH = 1;   // number of tiles in H dimension
    int tileCountW = 1;   // number of tiles in W dimension
    bool inplace = false; // indicates whether input and output buffers overlap

    // Network
    std::unique_ptr<Network> net;
    std::shared_ptr<Autoexposure> autoexposure;
    std::shared_ptr<InputProcess> inputProcess;
    std::shared_ptr<OutputProcess> outputProcess;
    std::shared_ptr<TransferFunction> transferFunc;
  };

} // namespace oidn
