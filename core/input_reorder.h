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
#include "image.h"

namespace oidn {

  // Input reorder node
  template<int K, class TransferFunction>
  class InputReorderNode : public Node
  {
  private:
    Image color;
    Image albedo;
    Image normal;

    std::shared_ptr<memory> dst;
    float* dstPtr;
    int C2;
    int H2;
    int W2;

    std::shared_ptr<TransferFunction> transferFunc;

  public:
    InputReorderNode(const Image& color,
                     const Image& albedo,
                     const Image& normal,
                     const std::shared_ptr<memory>& dst,
                     const std::shared_ptr<TransferFunction>& transferFunc)
      : color(color), albedo(albedo), normal(normal),
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
      assert(dstDesc.dims[2] >= color.height);
      assert(dstDesc.dims[3] >= color.width);

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
      const int H1 = color.height;
      const int W1 = color.width;

      // Do mirror padding to avoid filtering artifacts near the edges
      const int H = min(H2, 2*H1-2);
      const int W = min(W2, 2*W1-2);

      parallel_nd(H, [&](int h)
      {
        for (int w = 0; w < W; ++w)
        {
          // Compute mirror padded coords
          const int h1 = h < H1 ? h : 2*H1-2-h;
          const int w1 = w < W1 ? w : 2*W1-2-w;

          int c = 0;
          copyColor3(h, w, c, (float*)color.get(h1, w1));
          if (albedo)
            copy3<true>(h, w, c, (float*)albedo.get(h1, w1)); // clamp
          if (normal)
            copy3<false>(h, w, c, (float*)normal.get(h1, w1)); // don't clamp
        }
      });
    }

    std::shared_ptr<memory> getDst() const override { return dst; }

  private:
    // Stores a single value
    __forceinline void store(int h, int w, int& c, float value)
    {
      // Destination is in nChwKc format
      float* dst_c = dstPtr + (H2*W2*K*(c/K)) + h*W2*K + w*K + (c%K);
      *dst_c = value;
      c++;
    }

    // Copies 3 values with sanitization
    template<bool doClamp>
    __forceinline void copy3(int h, int w, int& c, const float* values)
    {
      store(h, w, c, sanitize<doClamp>(values[0]));
      store(h, w, c, sanitize<doClamp>(values[1]));
      store(h, w, c, sanitize<doClamp>(values[2]));
    }

    // Copies 3 color values with sanitization and applying the transfer function
    __forceinline void copyColor3(int h, int w, int& c, const float* values)
    {
      store(h, w, c, transferFunc->forward(sanitize<true>(values[0])));
      store(h, w, c, transferFunc->forward(sanitize<true>(values[1])));
      store(h, w, c, transferFunc->forward(sanitize<true>(values[2])));
    }
  };

} // namespace oidn
