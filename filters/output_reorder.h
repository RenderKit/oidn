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

#include "node.h"
#include "buffer_view.h"

namespace oidn {

  // Output reorder with optional conversion from sRGB to linear
  template<int K, bool srgb>
  class OutputReorder : public Node
  {
  private:
    std::shared_ptr<memory> src;
    const float* src_data;
    int H1;
    int W1;

    BufferView2D dst;

  public:
    OutputReorder(const std::shared_ptr<memory>& src, const BufferView2D& dst)
      : src(src), dst(dst)
    {
      memory::primitive_desc src_mpd = src->get_primitive_desc();
      const mkldnn_memory_desc_t& src_md = src_mpd.desc().data;
      assert(src_md.format == BlockedFormat<K>::nChwKc);
      assert(src_md.ndims == 4);
      assert(src_md.data_type == memory::data_type::f32);
      assert(src_md.dims[0] == 1);
      // We assume dst data is <= K OC
      assert(src_md.dims[1] == K);

      assert(dst.height <= src_md.dims[2]);
      assert(dst.width  <= src_md.dims[3]);

      src_data = (float*)src->get_data_handle();
      H1 = dst.height;
      W1 = dst.width;
    }

    void execute() override
    {
      const int C1 = K;
      const int H2 = dst.height;
      const int W2 = dst.width;

      tbb::parallel_for(tbb::blocked_range<int>(0, H2),
        [&](const tbb::blocked_range<int>& r)
        {
          for (int h = r.begin(); h != r.end(); ++h)
          {
            for (int w = 0; w < W2; ++w)
            {
              float* dst_C = (float*)dst.get(h, w);

              // Source is in nChwKc format. In this case C is 1 so this is really nhwc
              const float* src_C = src_data + h*W1*C1 + w*C1;

              if (srgb)
              {
                dst_C[0] = srgb_to_linear(src_C[0]);
                dst_C[1] = srgb_to_linear(src_C[1]);
                dst_C[2] = srgb_to_linear(src_C[2]);
              }
              else
              {
                dst_C[0] = src_C[0];
                dst_C[1] = src_C[1];
                dst_C[2] = src_C[2];
              }
            }
          }
        }, tbb::static_partitioner());
    }

  private:
    __forceinline float srgb_to_linear(float x)
    {
      return pow(x, 2.2f);
    }
  };

} // ::oidn
