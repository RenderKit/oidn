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
#include <cmath>

namespace oidn {

  // Input reorder with optional conversion from linear to sRGB
  template<int K, bool srgb>
  class InputReorder : public Node
  {
  private:
    BufferView2D src;
    BufferView2D src_albedo;
    BufferView2D src_normal;

    std::shared_ptr<memory> dst;
    float* dst_data;
    int C2;
    int H2;
    int W2;

    static constexpr int C1 = 9;

  public:
    InputReorder(const BufferView2D& src,
                 const BufferView2D& src_albedo,
                 const BufferView2D& src_normal,
                 const std::shared_ptr<memory>& dst)
      : src(src), src_albedo(src_albedo), src_normal(src_normal),
        dst(dst)
    {
      memory::primitive_desc dst_mpd = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& dst_md = dst_mpd.desc().data;
      assert(dst_md.format == BlockedFormat<K>::nChwKc);
      assert(dst_md.ndims == 4);
      assert(dst_md.data_type == memory::data_type::f32);
      assert(dst_md.dims[0] == 1);
      assert(dst_md.dims[1] >= padded<K>(C1));
      assert(dst_md.dims[2] >= src.height);
      assert(dst_md.dims[3] >= src.width);

      dst_data = (float*)dst->get_data_handle();
      C2 = dst_md.dims[1];
      H2 = dst_md.dims[2];
      W2 = dst_md.dims[3];

      // Zero the destination because it may be padded
      // We assume that the destination will not be modified by other nodes!
      memset(dst_data, 0, C2*H2*W2*sizeof(float));
    }

    void execute() override
    {
      const int H1 = src.height;
      const int W1 = src.width;

      tbb::parallel_for(tbb::blocked_range<int>(0, H1),
        [&](const tbb::blocked_range<int>& r)
        {
          for (int h = r.begin(); h != r.end(); ++h)
          {
            for (int w = 0; w < W1; ++w)
            {
              if (srgb)
                store3_srgb(h, w, 0, (float*)src.get(h, w));
              else
                store3(h, w, 0, (float*)src.get(h, w));

              store3(h, w, 3, (float*)src_albedo.get(h, w));
              store3(h, w, 6, (float*)src_normal.get(h, w));
            }
          }
        }, tbb::static_partitioner());
    }

    std::shared_ptr<memory> get_dst() const override { return dst; }

  private:
    __forceinline void store(int h, int w, int c, float value)
    {
      // Destination is in nChwKc format
      float* dst_c = dst_data + (H2*W2*K*(c/K)) + h*W2*K + w*K + (c%K);
      *dst_c = value;
    }

    __forceinline void store3(int h, int w, int c, const float* values)
    {
      store(h, w, c+0, values[0]);
      store(h, w, c+1, values[1]);
      store(h, w, c+2, values[2]);
    }

    __forceinline void store3_srgb(int h, int w, int c, const float* values)
    {
      store(h, w, c+0, linear_to_srgb(values[0]));
      store(h, w, c+1, linear_to_srgb(values[1]));
      store(h, w, c+2, linear_to_srgb(values[2]));
    }

    __forceinline float linear_to_srgb(float x)
    {
      return std::pow(x, 1.f/2.2f);
    }
  };

} // ::oidn
