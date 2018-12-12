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
      memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      memory::primitive_desc dstPrimDesc = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;
      const mkldnn_memory_desc_t& dstDesc = dstPrimDesc.desc().data;
      MAYBE_UNUSED(srcDesc);
      MAYBE_UNUSED(dstDesc);
      assert(srcDesc.format == BlockedFormat<K>::nChwKc);
      assert(dstDesc.format == BlockedFormat<K>::nChwKc);
      assert(srcDesc.ndims == 4);
      assert(dstDesc.ndims == 4);
      assert(srcDesc.data_type == memory::data_type::f32);
      assert(dstDesc.data_type == memory::data_type::f32);
      assert(srcDesc.dims[0] == 1);
      assert(dstDesc.dims[0] == 1);
      // 2x2 upsampling
      assert(dstDesc.dims[2] == srcDesc.dims[2] * 2);
      assert(dstDesc.dims[3] == srcDesc.dims[3] * 2);
    }

    void execute() override
    {
      memory::primitive_desc srcPrimDesc = src->get_primitive_desc();
      const mkldnn_memory_desc_t& srcDesc = srcPrimDesc.desc().data;

      const float* srcPtr = (float*)src->get_data_handle();
      float* dstPtr = (float*)dst->get_data_handle();

      const int C = srcDesc.dims[1];
      const int H = srcDesc.dims[2];
      const int W = srcDesc.dims[3];
      const int CK = C / K;

      parallel_nd(CK, H, [&](int ck, int h)
      {
        const size_t offset = ck*H*W*K + h*W*K;
        const float* srcPtr_line = srcPtr + offset;
        float* dstPtr_line0 = dstPtr + offset * 4;
        float* dstPtr_line1 = dstPtr_line0 + W*2*K; // next line

        for (int w = 0; w < W; ++w)
        {
          #pragma unroll
          for (int k = 0; k < K; k += 4)
          {
            const __m128 m = _mm_load_ps(&srcPtr_line[w*K + k]);

            _mm_stream_ps(&dstPtr_line0[w*2*K   + k], m);
            _mm_stream_ps(&dstPtr_line0[w*2*K+K + k], m);
            _mm_stream_ps(&dstPtr_line1[w*2*K   + k], m);
            _mm_stream_ps(&dstPtr_line1[w*2*K+K + k], m);
          }
        }
      });
    }

    std::shared_ptr<memory> getDst() const override { return dst; }
  };

} // namespace oidn
