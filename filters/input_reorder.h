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
    BufferView2D input;
    BufferView2D inputAlbedo;
    BufferView2D inputNormal;

    std::shared_ptr<memory> dst;
    float* dstPtr;
    int C2;
    int H2;
    int W2;

    std::shared_ptr<TransferFunction> transferFunc;

  public:
    InputReorder(const BufferView2D& input,
                 const BufferView2D& inputAlbedo,
                 const BufferView2D& inputNormal,
                 const std::shared_ptr<memory>& dst,
                 const std::shared_ptr<TransferFunction>& transferFunc)
      : input(input), inputAlbedo(inputAlbedo), inputNormal(inputNormal),
        dst(dst),
        transferFunc(transferFunc)
    {
      memory::primitive_desc dstPrimDesc = dst->get_primitive_desc();
      const mkldnn_memory_desc_t& dstDesc = dstPrimDesc.desc().data;
      assert(dstDesc.format == BlockedFormat<K>::nChwKc);
      assert(dstDesc.ndims == 4);
      assert(dstDesc.data_type == memory::data_type::f32);
      assert(dstDesc.dims[0] == 1);
      //assert(dstDesc.dims[1] >= getPadded<K>(C1));
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
      const int H1 = input.height;
      const int W1 = input.width;

      // Do mirror padding to avoid filtering artifacts near the edges
      const int H = std::min(H2, 2*H1-2);
      const int W = std::min(W2, 2*W1-2);

      tbb::parallel_for(tbb::blocked_range<int>(0, H),
        [&](const tbb::blocked_range<int>& r)
        {
          for (int h = r.begin(); h != r.end(); ++h)
          {
            for (int w = 0; w < W; ++w)
            {
              // Compute mirror padded coords
              const int h1 = h < H1 ? h : 2*H1-2-h;
              const int w1 = w < W1 ? w : 2*W1-2-w;

              int c = 0;
              storeColor3(h, w, c, (float*)input.get(h1, w1));
              if (inputAlbedo)
                store3(h, w, c, (float*)inputAlbedo.get(h1, w1));
              if (inputNormal)
                store3(h, w, c, (float*)inputNormal.get(h1, w1));
            }
          }
        }, tbb::static_partitioner());
    }

    std::shared_ptr<memory> getDst() const override { return dst; }

  private:
    __forceinline void store(int h, int w, int& c, float value)
    {
      // Destination is in nChwKc format
      float* dst_c = dstPtr + (H2*W2*K*(c/K)) + h*W2*K + w*K + (c%K);
      *dst_c = std::isfinite(value) ? value : 0.f; // filter out NaN and inf
      c++;
    }

    __forceinline void store3(int h, int w, int& c, const float* values)
    {
      store(h, w, c, values[0]);
      store(h, w, c, values[1]);
      store(h, w, c, values[2]);
    }

    __forceinline void storeColor3(int h, int w, int& c, const float* values)
    {
      store(h, w, c, transferFunc->forward(values[0]));
      store(h, w, c, transferFunc->forward(values[1]));
      store(h, w, c, transferFunc->forward(values[2]));
    }
  };

} // ::oidn
