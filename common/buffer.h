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

#include "ref.h"

namespace oidn {

  // Buffer which may or may not own its data
  class Buffer : public RefCount
  {
  private:
    char* ptr;
    size_t num_bytes;
    bool shared;

  public:
    __forceinline Buffer(size_t size)
      : ptr(new char[size]), num_bytes(size), shared(false) {}

    __forceinline Buffer(void* data, size_t size)
      : ptr((char*)data), num_bytes(size), shared(true) {}

    __forceinline ~Buffer()
    {
      if (!shared)
        delete[] ptr;
    }

    __forceinline char* data() { return ptr; }
    __forceinline const char* data() const { return ptr; }
    __forceinline size_t size() const { return num_bytes; }
  };

} // ::oidn
