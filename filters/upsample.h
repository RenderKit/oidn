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

namespace oidn {

  // 2x2 nearest-neighbor upsampling
  template<int K>
  class Upsample : public Node
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    Upsample(const std::shared_ptr<memory>& src, const std::shared_ptr<memory>& dst)
      : src(src), dst(dst)
    {
      memory::primitive_desc src_mpd = src->get_primitive_desc();
      memory::primitive_desc dst_mpd = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& src_md = src_mpd.desc().data;
      const mkldnn_memory_desc_t& dst_md = dst_mpd.desc().data;
      assert(src_md.format == BlockedFormat<K>::nChwKc);
      assert(dst_md.format == BlockedFormat<K>::nChwKc);
      assert(src_md.ndims == 4);
      assert(dst_md.ndims == 4);
      assert(src_md.data_type == memory::data_type::f32);
      assert(dst_md.data_type == memory::data_type::f32);
      assert(src_md.dims[0] == 1);
      assert(dst_md.dims[0] == 1);
      // 2x2 upsampling
      assert(dst_md.dims[2] == src_md.dims[2] * 2);
      assert(dst_md.dims[3] == src_md.dims[3] * 2);
    }

    void execute() override
    {
      memory::primitive_desc src_mpd = src->get_primitive_desc();
      memory::primitive_desc dst_mpd = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& src_md = src_mpd.desc().data;
      const mkldnn_memory_desc_t& dst_md = dst_mpd.desc().data;

      const float* src_data = (float*)src->get_data_handle();
      float* dst_data = (float*)dst->get_data_handle();

      const int C = src_md.dims[1];
      const int H = src_md.dims[2];
      const int W = src_md.dims[3];
      const int CK = C / K;

      const size_t work_amount = CK * H;
      tbb::parallel_for(tbb::blocked_range<size_t>(0, work_amount),
        [&](const tbb::blocked_range<size_t>& r)
        {
          int ck{0}, h{0};
          nd_iterator_init(r.begin(), ck, CK, h, H);

          for (size_t i = r.begin(); i != r.end(); ++i)
          {
            const size_t offset = ck*H*W*K + h*W*K;
            const float* src_line = src_data + offset;
            float* dst_line0 = dst_data + offset * 4;
            float* dst_line1 = dst_line0 + W*2*K; // next line

            for (int w = 0; w < W; ++w)
            {
              #pragma unroll
              for (int k = 0; k < K; k += 4)
              {
                const __m128 m = _mm_load_ps(&src_line[w*K + k]);

                _mm_stream_ps(&dst_line0[w*2*K   + k], m);
                _mm_stream_ps(&dst_line0[w*2*K+K + k], m);
                _mm_stream_ps(&dst_line1[w*2*K   + k], m);
                _mm_stream_ps(&dst_line1[w*2*K+K + k], m);
              }
            }

            nd_iterator_step(ck, CK, h, H);
          }
        }, tbb::static_partitioner());
    }

    std::shared_ptr<memory> get_dst() const override { return dst; }
  };

} // ::oidn
