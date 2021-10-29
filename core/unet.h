// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "filter.h"
#include "network.h"
#include "scratch.h"
#include "color.h"

namespace oidn {

  // ---------------------------------------------------------------------------
  // UNetFilter: U-Net based denoising filter
  // ---------------------------------------------------------------------------

  class UNetFilter : public Filter
  {
  protected:
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
    std::shared_ptr<InputReorderNode> inputReorder;
    std::shared_ptr<OutputReorderNode> outputReorder;
    std::shared_ptr<TransferFunction> transferFunc;

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
    } defaultWeights;
    Data userWeights;

    explicit UNetFilter(const Ref<Device>& device);
    virtual std::shared_ptr<TransferFunction> getTransferFunc() = 0;

  public:
    void setData(const std::string& name, const Data& data) override;
    void updateData(const std::string& name) override;
    void removeData(const std::string& name) override;
    void set1f(const std::string& name, float value) override;
    float get1f(const std::string& name) override;

    void commit() override;
    void execute(bool sync) override;

  private:
    void init();
    void computeTileSize();
    size_t buildNet(bool getScratchSizeOnly = false);
  };

  // ---------------------------------------------------------------------------
  // RTFilter: Generic ray tracing denoiser
  // ---------------------------------------------------------------------------

  class RTFilter : public UNetFilter
  {
  public:
    explicit RTFilter(const Ref<Device>& device);

    void setImage(const std::string& name, const std::shared_ptr<Image>& image) override;
    void removeImage(const std::string& name) override;
    void set1i(const std::string& name, int value) override;
    int get1i(const std::string& name) override;
  
  protected:
    std::shared_ptr<TransferFunction> getTransferFunc() override;
  };

  // ---------------------------------------------------------------------------
  // RTLightmapFilter: Ray traced lightmap denoiser
  // ---------------------------------------------------------------------------

  class RTLightmapFilter : public UNetFilter
  {
  public:
    explicit RTLightmapFilter(const Ref<Device>& device);

    void setImage(const std::string& name, const std::shared_ptr<Image>& image) override;
    void removeImage(const std::string& name) override;
    void set1i(const std::string& name, int value) override;
    int get1i(const std::string& name) override;

  protected:
    std::shared_ptr<TransferFunction> getTransferFunc() override;
  };

} // namespace oidn
