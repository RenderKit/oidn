// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include "sycl_conv_dpas.h"

namespace oidn {

  using namespace esimd;
  using namespace sycl::ext::intel::experimental::esimd;

  constexpr int owBlock = 16;
  constexpr int iwBlock = owBlock + 3 - 1;

  constexpr int ohBlock = 3;

  template<typename T, TensorLayout tensorLayout, TensorLayout weightLayout>
  struct SYCLConvDPASKernel
  {
    using AccumType = float;

    static constexpr int execWidth  = 8;
    static constexpr int dpasDepth  = 8;
    static constexpr int dpasRepeat = 8;

    static constexpr int cBlock = TensorAccessor3D<T, tensorLayout>::cBlock;
    static constexpr int ocSubblock = execWidth;
    static constexpr int ocOuter = cBlock / ocSubblock;

    static constexpr int owSubblock = dpasRepeat;
    static constexpr int owOuter = owBlock / owSubblock;
    
    TensorAccessor3D<T, tensorLayout> src;
    TensorAccessor4D<T, weightLayout> weight;
    TensorAccessor1D<T> bias;
    TensorAccessor3D<T, tensorLayout> dst;

    OIDN_INLINE void operator ()(const WorkItem<3>& it) const SYCL_ESIMD_FUNCTION
    {
      set_kernel_properties(kernel_properties::use_double_grf);

      const int oc = it.getId<0>() * cBlock;
      const int oh = it.getId<1>() * ohBlock;
      const int ow = it.getId<2>() * owBlock;

      // Accumulators
      simd<AccumType, owSubblock * ocSubblock> accumVec[ohBlock][owOuter][ocOuter] = {}; // = 0

      // Iterate over input channel blocks
      for (int ic = 0; ic < src.C; ic += cBlock)
      {
        const int iw = ow - 1;

        simd<T, iwBlock*cBlock> srcVec[ohBlock];
        #pragma unroll
        for (int r = 0; r < ohBlock - 1; ++r)
          loadRow(srcVec[r], ic, oh + r - 1, iw);

        // Iterate over kernel height
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh)
        {
          // Load next input row
          loadRow(srcVec[(kh + ohBlock - 1) % ohBlock], ic, oh + ohBlock - 2 + kh, iw);

          // Iterate over kernel width
          const T* weightPtr = &weight(oc, ic, kh, 0);
          
          #pragma unroll
          for (int kw = 0; kw < 3; ++kw)
          {
            // Load weights
            simd<T, ocSubblock*cBlock> weightVec[ocOuter];
            #pragma unroll
            for (int i = 0; i < ocOuter; ++i)
            {
              weightVec[i].copy_from(weightPtr, vector_aligned);
              weightPtr += ocSubblock*cBlock;
            }

            // Multiply + accumulate
            #pragma unroll
            for (int r = 0; r < ohBlock; ++r)
            {
              #pragma unroll
              for (int j = 0; j < owOuter; ++j)
              {
                #pragma unroll
                for (int i = 0; i < ocOuter; ++i)
                {
                  accumVec[r][j][i] =
                    dpas<argument_type::FP16,
                        argument_type::FP16,
                        AccumType,
                        dpasDepth,
                        dpasRepeat,
                        AccumType,
                        int,
                        int,
                        dpasRepeat * execWidth,
                        dpasDepth  * execWidth,
                        dpasRepeat * dpasDepth
                        >(accumVec[r][j][i],
                          weightVec[i].template bit_cast_view<int>(),
                          srcVec[(kh + r) % ohBlock].template select<owSubblock*cBlock, 1>((kw+j*owSubblock)*cBlock).template bit_cast_view<int>());
                }
              }
            }
          }
        }
      }

      const auto biasVec = block_load<T, cBlock>(&bias(oc), vector_aligned);
      
      #pragma unroll
      for (int r = 0; r < ohBlock; ++r)
      {
        if (oh + r >= dst.H)
          continue;

        T* dstPtr = &dst(oc, oh + r, ow);

        #pragma unroll
        for (int j = 0; j < owOuter; ++j)
        {
          // Shuffle and convert accumulators
          simd<T, owSubblock*cBlock> dstVec;
          auto dstMat = dstVec.template bit_cast_view<T, owSubblock, cBlock>();
          #pragma unroll
          for (int i = 0; i < ocOuter; ++i)
            dstMat.template select<owSubblock, 1, ocSubblock, 1>(0, i*ocSubblock) = accumVec[r][j][i];

          // Add bias
          dstVec += biasVec.template replicate<owSubblock>();

          // Apply ReLU
          dstVec = max(dstVec, simd<T, owSubblock*cBlock>(0));

          // Store output row
          #pragma unroll
          for (int i = 0; i < owSubblock; ++i)
          {
            if (ow + j*owSubblock + i < dst.W)
              block_store<T, cBlock>(dstPtr, dstVec.template select<cBlock, 1>(i * cBlock));
            dstPtr += cBlock;
          }
        }
      }
    }

    OIDN_INLINE void loadRow(simd<T, iwBlock*cBlock>& srcVec, int ic, int ih, int iw) const
    {
      if (ih < 0 || ih >= src.H)
      {
        srcVec = 0;
        return;
      }

      const T* srcPtr = &src(ic, ih, iw);

      if (iw >= 0 && iw + iwBlock < src.W)
      {
        srcVec.copy_from(srcPtr, overaligned<32>);
      }
      else
      {
        srcVec = 0;
        #pragma unroll
        for (int i = 0; i < iwBlock; ++i)
        {
          if (iw + i >= 0 && iw + i < src.W)
            srcVec.template select<cBlock, 1>(i*cBlock) = block_load<T, cBlock>(srcPtr, vector_aligned);
          srcPtr += cBlock;
        }
      }
    }
  };

  SYCLConvDPAS::SYCLConvDPAS(const Ref<SYCLDevice>& device, const ConvDesc& desc)
    : Conv(desc),
      device(device)
  {
    if (srcDesc.layout != TensorLayout::Chw16c || srcDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution source layout/data type");
    if (weightDesc.layout != TensorLayout::OIhw2o8i8o2i || weightDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution weight layout/data type");
    if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataType::Float16)
      throw std::invalid_argument("unsupported convolution bias layout/data type");
  }

  void SYCLConvDPAS::run()
  {
    if (!src || !weight || !bias || !dst)
      throw std::logic_error("convolution argument not set");

    SYCLConvDPASKernel<half, TensorLayout::Chw16c, TensorLayout::OIhw2o8i8o2i> kernel;
    kernel.src    = *src;
    kernel.weight = *weight;
    kernel.bias   = *bias;
    kernel.dst    = *dst;

    // FIXME: need to round up WB dimension to multiple of 2 due to DPAS bug
    device->runESIMDKernelAsync(WorkDim<3>(dst->getCB(), ceil_div(dst->getH(), ohBlock), round_up(ceil_div(dst->getW(), owBlock), 2)), kernel);
  }

} // namespace oidn