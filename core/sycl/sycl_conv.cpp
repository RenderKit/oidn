// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_conv.h"

namespace oidn {

  constexpr int owBlock = 16;
  constexpr int iwBlock = owBlock + 3 - 1;

  template<typename T, TensorLayout tensorLayout, TensorLayout weightLayout>
  struct SYCLConvKernel
  {
    static constexpr int B = TensorAccessor3D<T, tensorLayout>::B;
    static constexpr int V = 128 / sizeof(T);

    TensorAccessor3D<T, tensorLayout> src;
    TensorAccessor4D<T, weightLayout> weight;
    TensorAccessor1D<T> bias;
    TensorAccessor3D<T, tensorLayout> dst;

    OIDN_INLINE void operator ()(const WorkItem<3>& it) const SYCL_ESIMD_FUNCTION
    { 
      using namespace esimd;

      const int oc = it.getId<0>() * B;
      const int oh = it.getId<1>();
      const int ow = it.getId<2>() * owBlock;

      simd<T, B> accum[owBlock];
      const auto b = block_load<T, B, vector_aligned_tag>(&bias(oc));
      #pragma unroll
      for (int i = 0; i < owBlock; ++i)
        accum[i] = b;

      for (int ic = 0; ic < src.C; ic += B)
      {
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh)
        {
          const int ih = oh + kh - 1;
          if (ih < 0 || ih >= src.H)
            continue;

          const int iw = ow - 1;
          const T* srcPtr = &src(ic, ih, iw);
          simd<T, B> a[iwBlock];
          #pragma unroll
          for (int i = 0; i < iwBlock; ++i)
          {
            if (iw + i < 0 || iw + i >= src.W)
              a[i] = 0;
            else
              a[i] = block_load<T, B, vector_aligned_tag>(srcPtr);
            srcPtr += B;
          }

          #pragma unroll
          for (int kw = 0; kw < 3; ++kw)
          {
            const T* weightPtr = &weight(oc, ic, kw, kh);
            simd<T, B*B> w;

            #pragma unroll
            for (int i = 0; i < B*B; i += V)
            {
              w.template select<V, 1>(i) = block_load<T, V, vector_aligned_tag>(weightPtr);
              weightPtr += V;
            }

            #pragma unroll
            for (int i = 0; i < B; ++i)
            {
              #pragma unroll
              for (int j = 0; j < owBlock; ++j)
                accum[j] += a[j+kw].template replicate_w<B, 1>(i) * w.template select<B, 1>(i * B);
            }
          }
        }
      }

      #pragma unroll
      for (int i = 0; i < owBlock; ++i)
        accum[i] = max(accum[i], simd<T, B>(0));

      T* dstPtr = &dst(oc, oh, ow);
      #pragma unroll
      for (int i = 0; i < owBlock; ++i)
      {
        if (ow + i < dst.W)
          block_store(dstPtr, accum[i]);
        dstPtr += B;
      }
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

    device->runESIMDKernelAsync(WorkDim<3>(dst->getCB(), dst->getH(), ceil_div(dst->getW(), owBlock)), kernel);
  }

} // namespace oidn