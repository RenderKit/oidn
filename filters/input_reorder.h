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

  // Input reorder
  template<int K, class TransferFunction>
  class InputReorder : public Node
  {
  private:
    BufferView2D src;
    BufferView2D srcAlbedo;
    BufferView2D srcNormal;

    std::shared_ptr<memory> dst;
    float* dstPtr;
    int C2;
    int H2;
    int W2;

    TransferFunction transfer;

    static constexpr int C1 = 9;

  public:
    InputReorder(const BufferView2D& src,
                 const BufferView2D& srcAlbedo,
                 const BufferView2D& srcNormal,
                 const std::shared_ptr<memory>& dst,
                 const TransferFunction& transfer)
      : src(src), srcAlbedo(srcAlbedo), srcNormal(srcNormal),
        dst(dst),
        transfer(transfer)
    {
      memory::primitive_desc dstPrimDesc = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& dstDesc = dstPrimDesc.desc().data;
      assert(dstDesc.format == BlockedFormat<K>::nChwKc);
      assert(dstDesc.ndims == 4);
      assert(dstDesc.data_type == memory::data_type::f32);
      assert(dstDesc.dims[0] == 1);
      assert(dstDesc.dims[1] >= getPadded<K>(C1));
      assert(dstDesc.dims[2] >= src.height);
      assert(dstDesc.dims[3] >= src.width);

      dstPtr = (float*)dst->get_data_handle();
      C2 = dstDesc.dims[1];
      H2 = dstDesc.dims[2];
      W2 = dstDesc.dims[3];

      // Zero the destination because it may be padded
      // We assume that the destination will not be modified by other nodes!
      memset(dstPtr, 0, C2*H2*W2*sizeof(float));
    }

    void execute() override
    {
      const int H1 = src.height;
      const int W1 = src.width;

      tbb::parallel_for(tbb::blocked_range<int>(0, H2),
        [&](const tbb::blocked_range<int>& r)
        {
          for (int h = r.begin(); h != r.end(); ++h)
          {
            for (int w = 0; w < W2; ++w)
            {
              // Mirror padding to avoid filtering artifacts near the edges
              int hm = h < H1 ? h : 2*H1-h-2;
              int wm = w < W1 ? w : 2*W1-w-2;

              store3Transfer(h, w, 0, (float*)src.get(hm, wm));
              store3(h, w, 3, (float*)srcAlbedo.get(hm, wm));
              store3(h, w, 6, (float*)srcNormal.get(hm, wm));
            }
          }
        }, tbb::static_partitioner());
    }

    std::shared_ptr<memory> getDst() const override { return dst; }

  private:
    __forceinline void store(int h, int w, int c, float value)
    {
      // Destination is in nChwKc format
      float* dst_c = dstPtr + (H2*W2*K*(c/K)) + h*W2*K + w*K + (c%K);
      *dst_c = std::isfinite(value) ? value : 0.f; // filter out NaN and inf
    }

    __forceinline void store3(int h, int w, int c, const float* values)
    {
      store(h, w, c+0, values[0]);
      store(h, w, c+1, values[1]);
      store(h, w, c+2, values[2]);
    }

    __forceinline void store3Transfer(int h, int w, int c, const float* values)
    {
      store(h, w, c+0, transfer.forward(values[0]));
      store(h, w, c+1, transfer.forward(values[1]));
      store(h, w, c+2, transfer.forward(values[2]));
    }
  };

} // ::oidn
