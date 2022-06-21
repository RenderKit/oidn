// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_conv.h"

namespace oidn {

  template<typename T, TensorLayout tensorLayout, TensorLayout weightLayout>
  struct SYCLConvKernel
  {
    static constexpr int B = TensorAccessor3D<T, tensorLayout>::B;

    TensorAccessor3D<T, tensorLayout> src;
    TensorAccessor4D<T, weightLayout> weight;
    TensorAccessor1D<T> bias;
    TensorAccessor3D<T, tensorLayout> dst;

    OIDN_INLINE void operator ()(const WorkItem<3>& it) const SYCL_ESIMD_FUNCTION
    { 
      using namespace esimd;

      const int oc = it.getId<0>() * B;
      const int oh = it.getId<1>();
      const int ow = it.getId<2>();

      simd<T, B> accum = block_load<T, B, vector_aligned_tag>(&bias(oc));

      for (int ic = 0; ic < src.C; ic += B)
      {
        simd<T, B> w[B];

        for (int kh = 0; kh < 3; ++kh)
        {
          const int ih = oh + kh - 1;
          if (ih >= 0 && ih < src.H)
          {
            for (int kw = 0; kw < 3; ++kw)
            {
              const int iw = ow + kw - 1;
              if (iw >= 0 && iw < src.W)
              {
                #pragma unroll
                for (int i = 0; i < B; ++i)
                  w[i] = block_load<T, B, vector_aligned_tag>(&weight(oc, ic + i, kw, kh));

                #pragma unroll
                for (int i = 0; i < B; ++i)
                {
                  const simd<T, B> a = block_load<T, B, vector_aligned_tag>(&src(ic, ih, iw));
                  accum += a.template replicate_w<B, 1>(i) * w[i];
                }
              }
            }
          }
        }
      }

      accum = max(accum, simd<T, B>(0));
      block_store(&dst(oc, oh, ow), accum);
    }
  };

  SYCLConv::SYCLConv(const Ref<SYCLDevice>& device, const ConvDesc& desc)
    : Conv(desc),
      device(device)
  {
    if (srcDesc.layout != TensorLayout::Chw16c || srcDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution source layout/data type");
    if (weightDesc.layout != TensorLayout::OIhw16i16o || weightDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution weight layout/data type");
    if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution bias layout/data type");
  }

  void SYCLConv::run()
  {
    if (!src || !weight || !bias || !dst)
      throw std::logic_error("convolution argument not set");

    SYCLConvKernel<half, TensorLayout::Chw16c, TensorLayout::OIhw16i16o> kernel;
    kernel.src    = *src;
    kernel.weight = *weight;
    kernel.bias   = *bias;
    kernel.dst    = *dst;

    device->runESIMDKernelAsync(WorkDim<3>(dst->getCB(), dst->getH(), dst->getW()), kernel);
  }

} // namespace oidn