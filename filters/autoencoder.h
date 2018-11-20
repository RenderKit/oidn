// ======================================================================== //
// Copyright 2009-2018 Intel Corporation                                    //
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

namespace oidn {

  // Direct-predicting autoencoder
  class Autoencoder : public Filter
  {
  private:
    BufferView2D input;
    BufferView2D inputAlbedo;
    BufferView2D inputNormal;
    BufferView2D output;
    bool srgb;
    bool hdr;

    std::shared_ptr<Node> net;

  public:
    Autoencoder(const Ref<Device>& device);

    void setBuffer2D(const std::string& name, int slot, const BufferView2D& view) override;
    void set1i(const std::string& name, int value) override;
    void commit() override;
    void execute() override;

  private:
    template<int K>
    std::shared_ptr<Node> buildNet();
  };

} // ::oidn
