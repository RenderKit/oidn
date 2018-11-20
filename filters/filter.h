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

#include "common.h"
#include "device.h"
#include "buffer_view.h"

namespace oidn {

  class Filter : public RefCount
  {
  protected:
    Ref<Device> device;

  public:
    explicit Filter(const Ref<Device>& device) : device(device) {}

    virtual void setBuffer2D(const std::string& name, int slot, const BufferView2D& view) = 0;
    virtual void set1i(const std::string& name, int value) {};
    virtual void commit() = 0;
    virtual void execute() = 0;

    Ref<Device> getDevice() const { return device; }
  };

} // ::oidn
