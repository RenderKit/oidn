// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "node.h"
#include "upsample_ispc.h"

namespace oidn {

#if defined(OIDN_DNNL)

  // 2x2 nearest-neighbor upsampling node (blocked layout)
  class UpsampleNode : public Node
  {
  private:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;
    int K;

  public:
    UpsampleNode(const Ref<Device>& device,
                 const std::shared_ptr<Tensor>& src,
                 const std::shared_ptr<Tensor>& dst)
      : Node(device),
        src(src),
        dst(dst)
    {
      assert(src->ndims() == 3);
      assert(src->layout == TensorLayout::Chw8c ||
             src->layout == TensorLayout::Chw16c);
      assert(src->blockSize() == device->getTensorBlockSize());
      assert(dst->ndims() == 3);
      assert(dst->layout == src->layout);
      assert(dst->dims[0] == src->dims[0]);     // C
      assert(dst->dims[1] == src->dims[1] * 2); // H
      assert(dst->dims[2] == src->dims[2] * 2); // W

      K = device->getTensorBlockSize();
    }

    void execute() override
    {
      ispc::Upsample impl;
      impl.src = *src;
      impl.dst = *dst;

      parallel_nd(impl.src.C / K, impl.src.H, [&](int ck, int h)
      {
        ispc::Upsample_kernel(&impl, ck, h);
      });
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

#else

  // 2x2 nearest-neighbor upsampling node
  class UpsampleNode : public Node
  {
  private:
    std::shared_ptr<Tensor> src;
    std::shared_ptr<Tensor> dst;

  public:
    UpsampleNode(const Ref<Device>& device,
                 const std::shared_ptr<Tensor>& src,
                 const std::shared_ptr<Tensor>& dst)
      : Node(device),
        src(src),
        dst(dst)
    {
      assert(src->ndims() == 3);
      assert(src->layout == TensorLayout::chw);
      assert(dst->ndims() == 3);
      assert(dst->layout == src->layout);
      assert(dst->dims[0] == src->dims[0]);     // C
      assert(dst->dims[1] == src->dims[1] * 2); // H
      assert(dst->dims[2] == src->dims[2] * 2); // W
    }

    void execute() override
    {
      const size_t C = src->dims[0];
      const size_t H = src->dims[1];
      const size_t W = src->dims[2];

      parallel_nd(C, H, [&](int c, int h)
      {
        const size_t offset = (c*H + h) * W;
        const float* srcPtr_line = (float*)src->data() + offset;
        float* dstPtr_line0 = (float*)dst->data() + offset * 4;
        float* dstPtr_line1 = dstPtr_line0 + W*2; // next line

        #pragma unroll(16)
        for (size_t w = 0; w < W; ++w)
        {
          // Load value
          const float value = srcPtr_line[w];

          // Store value 2x2
          dstPtr_line0[w*2  ] = value;
          dstPtr_line0[w*2+1] = value;
          dstPtr_line1[w*2  ] = value;
          dstPtr_line1[w*2+1] = value;
        }
      });
    }

    std::shared_ptr<Tensor> getDst() const override { return dst; }
  };

#endif

} // namespace oidn
