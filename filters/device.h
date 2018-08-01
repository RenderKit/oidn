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

namespace oidn {

  class Buffer;
  class Filter;

  class Device : public RefCount
  {
  private:
    // Tasking
    std::shared_ptr<tbb::task_arena> arena;
    std::shared_ptr<PinningObserver> observer;
    std::shared_ptr<ThreadAffinity> affinity;

  public:
    Device();
    ~Device();

    template<typename F>
    void execute_task(F& f)
    {
      arena->execute(f);
    }

    template<typename F>
    void execute_task(const F& f)
    {
      arena->execute(f);
    }

    Ref<Buffer> new_buffer(void* ptr, size_t byte_size);
    Ref<Filter> new_filter(FilterType type);
  };

} // ::oidn
