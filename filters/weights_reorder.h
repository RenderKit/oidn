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

  // Reorders weights from oihw to OIhwKiKo format
  template<int K>
  class WeightsReorder : public Node
  {
  private:
    std::shared_ptr<memory> src;
    std::shared_ptr<memory> dst;

  public:
    WeightsReorder(const std::shared_ptr<memory>& src, const std::shared_ptr<memory>& dst)
      : src(src), dst(dst)
    {
      memory::primitive_desc src_mpd = src->get_primitive_desc();
      memory::primitive_desc pad_mpd = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& src_md = src_mpd.desc().data;
      const mkldnn_memory_desc_t& pad_md = pad_mpd.desc().data;
      assert(src_md.format == memory::format::oihw);
      assert(pad_md.format == BlockedFormat<K>::OIhwKiKo);
      assert(src_md.ndims == 4);
      assert(pad_md.ndims == 4);
      assert(src_md.data_type == memory::data_type::f32);
      assert(pad_md.data_type == memory::data_type::f32);
      assert(padded<K>(src_md.dims[0]) == pad_md.dims[0]); // OC
      assert(padded<K>(src_md.dims[1]) == pad_md.dims[1]); // IC
      assert(src_md.dims[2] == pad_md.dims[2]);
      assert(src_md.dims[3] == pad_md.dims[3]);
    }

    void execute() override
    {
      memory::primitive_desc src_mpd = src->get_primitive_desc();
      memory::primitive_desc pad_mpd = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& src_md = src_mpd.desc().data;
      const mkldnn_memory_desc_t& pad_md = pad_mpd.desc().data;

      const float* src_data = (float*)src->get_data_handle();
      float* dst_data = (float*)dst->get_data_handle();

      const int OC1 = src_md.dims[0];
      const int OC2 = pad_md.dims[0];
      const int IC1 = src_md.dims[1];
      const int IC2 = pad_md.dims[1];
      const int H   = pad_md.dims[2];
      const int W   = pad_md.dims[3];

      for (int oc = 0; oc < OC2; ++oc)
      {
        for (int ic = 0; ic < IC2; ++ic)
        {
          for (int h = 0; h < H; ++h)
          {
            for (int w = 0; w < W; ++w)
            {
              // Output is in OIhwKiKo format
              float* dst_c = dst_data + (oc/K)*(IC2/K)*H*W*K*K +
                                        (ic/K)*H*W*K*K +
                                        h*W*K*K +
                                        w*K*K +
                                        (ic%K)*K + (oc%K);

              if (oc < OC1 && ic < IC1)
              {
                // Input is in oihw format
                const float* src_c = src_data + oc*IC1*H*W + ic*H*W + h*W + w;
                *dst_c = *src_c;
              }
              else
              {
                // padding
                *dst_c = 0;
              }
            }
          }
        }
      }
    }

    std::shared_ptr<memory> get_dst() const override { return dst; }
  };

} // ::oidn
