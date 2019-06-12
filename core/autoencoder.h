// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include "filter.h"
#include "network.h"
#include "tone_mapping.h"

namespace oidn {

  // --------------------------------------------------------------------------
  // AutoencoderFilter - Direct-predicting autoencoder
  // --------------------------------------------------------------------------

  class AutoencoderFilter : public Filter
  {
  private:
    static constexpr int padding = 32;  // the image must be padded spatially
    static constexpr int overlap = 128; // amount of overlap between tiles (rounded up to the padding size)

    Image color;
    Image albedo;
    Image normal;
    Image output;
    bool hdr = false;
    bool srgb = false;

    int H = 0;          // image height
    int W = 0;          // image width
    int tileH = 0;      // tile height
    int tileW = 0;      // tile width
    int tileCountH = 1; // number of tiles in H dimension
    int tileCountW = 1; // number of tiles in W dimension

    std::shared_ptr<Executable> net;
    std::shared_ptr<Node> inputReorder;
    std::shared_ptr<Node> outputReorder;
    std::shared_ptr<TransferFunction> transferFunc;

  protected:
    struct
    {
      void* ldr         = nullptr;
      void* ldr_alb     = nullptr;
      void* ldr_alb_nrm = nullptr;
      void* hdr         = nullptr;
      void* hdr_alb     = nullptr;
      void* hdr_alb_nrm = nullptr;
    } weightData;

    explicit AutoencoderFilter(const Ref<Device>& device);

  public:
    void setImage(const std::string& name, const Image& data) override;
    void set1i(const std::string& name, int value) override;
    int get1i(const std::string& name) override;

    void commit() override;
    void execute() override;

  private:
    void computeTileSize();

    template<int K>
    std::shared_ptr<Executable> buildNet();

    bool isCommitted() const { return bool(net); }
  };

  // --------------------------------------------------------------------------
  // RTFilter - Generic ray tracing denoiser
  // --------------------------------------------------------------------------

  class RTFilter : public AutoencoderFilter
  {
  public:
    explicit RTFilter(const Ref<Device>& device);
  };

} // namespace oidn
