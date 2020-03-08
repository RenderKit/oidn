// =============================================================================
// Copyright 2009-2020 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#pragma once

#include "node.h"
#include "upsample_ispc.h"

namespace oidn {

  // 2x2 nearest-neighbor upsampling node
  class UpsampleNode : public Node
  {
  private:
    ispc::Upsample data;

    int K;
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    UpsampleNode(int K,
                 const std::shared_ptr<memory>& src,
                 const std::shared_ptr<memory>& dst)
      : K(K),
        src(src),
        dst(dst)
    {
      data.src = toIspc(src);
      data.dst = toIspc(dst);

      assert(data.dst.H == data.src.H * 2);
      assert(data.dst.W == data.src.W * 2);
    }

    void execute(stream& sm) override
    {
      parallel_nd(data.src.C / K, data.src.H, [&](int ck, int h)
      {
        ispc::Upsample_kernel(&data, ck, h);
      });
    }

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

} // namespace oidn
