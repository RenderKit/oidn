// Copyright 2009-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "filter.h"
#include "network.h"
#include "color.h"

namespace oidn {

  // ---------------------------------------------------------------------------
  // UNetFilter: U-Net based denoising filter
  // ---------------------------------------------------------------------------

  class UNetFilter : public Filter
  {
  protected:
    // Network constants
    static constexpr int alignment       = 32;  // required spatial alignment in pixels (padding may be necessary)
    static constexpr int receptiveField  = 222; // receptive field in pixels
    static constexpr int overlap         = round_up(receptiveField / 2, alignment); // required spatial overlap between tiles in pixels

    // Estimated memory usage
    static constexpr int estimatedBytesBase     = 16*1024*1024; // conservative base memory usage
    static constexpr int estimatedBytesPerPixel = 889;

    // Images
    Image color;
    Image albedo;
    Image normal;
    Image output;
    Image outputTemp; // required for in-place tiled filtering

    // Options
    bool hdr = false;
    float hdrScale = std::numeric_limits<float>::quiet_NaN();
    bool srgb = false;
    int maxMemoryMB = 6000; // approximate maximum memory usage in MBs

    // Image dimensions
    int H = 0;            // image height
    int W = 0;            // image width
    int tileH = 0;        // tile height
    int tileW = 0;        // tile width
    int tileCountH = 1;   // number of tiles in H dimension
    int tileCountW = 1;   // number of tiles in W dimension
    bool inplace = false; // indicates whether input and output buffers overlap

    // Network
    std::shared_ptr<Executable> net;
    std::shared_ptr<Node> inputReorder;
    std::shared_ptr<Node> outputReorder;

    // Weights
    struct
    {
      Data ldr;
      Data ldr_alb;
      Data ldr_alb_nrm;
      Data hdr;
      Data hdr_alb;
      Data hdr_alb_nrm;
    } defaultWeights;
    Data userWeights;

    explicit UNetFilter(const Ref<Device>& device);
    virtual std::shared_ptr<TransferFunction> makeTransferFunc();

  public:
    void setImage(const std::string& name, const Image& data) override;
    void setData(const std::string& name, const Data& data) override;
    void set1i(const std::string& name, int value) override;
    int get1i(const std::string& name) override;
    void set1f(const std::string& name, float value) override;
    float get1f(const std::string& name) override;

    void commit() override;
    void execute() override;

  private:
    void computeTileSize();

    std::shared_ptr<Executable> buildNet();

    bool isCommitted() const { return bool(net); }
  };

  // ---------------------------------------------------------------------------
  // RTFilter: Generic ray tracing denoiser
  // ---------------------------------------------------------------------------

  class RTFilter : public UNetFilter
  {
  public:
    explicit RTFilter(const Ref<Device>& device);
  };

  // ---------------------------------------------------------------------------
  // RTLightmapFilter: Ray traced lightmap denoiser
  // ---------------------------------------------------------------------------

  class RTLightmapFilter : public UNetFilter
  {
  public:
    explicit RTLightmapFilter(const Ref<Device>& device);
    std::shared_ptr<TransferFunction> makeTransferFunc() override;
  };

} // namespace oidn
