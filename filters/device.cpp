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

#include "device.h"
#include "autoencoder.h"

namespace oidn {

  Device::Device()
  {
    affinity = std::make_shared<ThreadAffinity>(1); // one thread per core
    arena = std::make_shared<tbb::task_arena>(affinity->num_threads());
    observer = std::make_shared<PinningObserver>(affinity, *arena);
  }

  Device::~Device()
  {
    observer.reset();
  }

  Ref<Buffer> Device::new_buffer(void* ptr, size_t byte_size)
  {
    return make_ref<Buffer>(ptr, byte_size);
  }

  Ref<Filter> Device::new_filter(const std::string& type)
  {
    Ref<Filter> filter;

    if (type == "Autoencoder")
      filter = make_ref<Autoencoder>(Ref<Device>(this));
    else
      throw std::invalid_argument("invalid filter type");

    return filter;
  }

} // ::oidn
