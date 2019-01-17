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

#include "common.h"

namespace oidn {

  class Buffer;
  class Filter;

  class Device : public RefCount
  {
  private:
    // Error handling
    static thread_local Error threadError;
    static thread_local std::string threadErrorMessage;
    Error error;
    std::string errorMessage;

    // Tasking
    std::shared_ptr<tbb::task_arena> arena;
    std::shared_ptr<PinningObserver> observer;
    std::shared_ptr<ThreadAffinity> affinity;

  public:
    Device();
    ~Device();

    static void setError(Device* device, Error error, const std::string& errorMessage);
    static Error getError(Device* device, const char** errorMessage);

    int get1i(const std::string& name);

    void commit();

    template<typename F>
    void executeTask(F& f)
    {
      arena->execute(f);
    }

    template<typename F>
    void executeTask(const F& f)
    {
      arena->execute(f);
    }

    Ref<Buffer> newBuffer(size_t byteSize);
    Ref<Buffer> newBuffer(void* ptr, size_t byteSize);
    Ref<Filter> newFilter(const std::string& type);

    Device* getDevice() { return this; }

  private:
    bool isCommitted() const { return bool(arena); }

    void checkCommitted();
  };

} // namespace oidn
