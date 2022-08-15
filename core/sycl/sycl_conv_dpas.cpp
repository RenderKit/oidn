// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include "sycl_conv_dpas.h"

namespace oidn {

  using namespace esimd;
  using namespace sycl::ext::intel::experimental::esimd;

  template<typename T, TensorLayout tensorLayout, TensorLayout weightLayout>
  struct SYCLConvDPASKernel
  {
    using AT = float; // accumulator type

    static constexpr int execWidth  = 8; // SIMD execution width
    static constexpr int dpasDepth  = 8; // DPAS depth
    static constexpr int dpasRepeat = 8; // DPAS repeat count

    static constexpr int blockOH = 5;               // block output height
    static constexpr int blockOW = dpasRepeat;      // block output width
    static constexpr int blockIW = blockOW + 3 - 1; // block input width

    static constexpr int blockC = TensorAccessor3D<T, tensorLayout>::blockC; // block input/output channels
    static constexpr int blockAC = execWidth;           // block accumulator channels
    static constexpr int numBlockAC = blockC / blockAC; // number of accumulator channel blocks
    
    TensorAccessor3D<T, tensorLayout> src;
    TensorAccessor4D<T, weightLayout> weight;
    TensorAccessor1D<T> bias;
    TensorAccessor3D<T, tensorLayout> dst;

    OIDN_INLINE void operator ()(const WorkGroupItem<3>& it) const SYCL_ESIMD_FUNCTION
    {
      set_kernel_properties(kernel_properties::use_double_grf);

      const int oc = it.getLocalId<0>()  * blockC;
      const int oh = it.getGlobalId<1>() * blockOH;
      const int ow = it.getGlobalId<2>() * blockOW;

      // Accumulators
      simd<AT, blockOW * blockAC> accumVec[blockOH][numBlockAC] = {}; // = 0

      // Iterate over input channel blocks
      for (int ic = 0; ic < src.C; ic += blockC)
      {
        const int ih = oh - 1;
        const int iw = ow - 1;

        // Load input rows into ring buffer
        simd<T, blockIW * blockC> srcVec[blockOH];
        #pragma unroll
        for (int boh = 0; boh < blockOH - 1; ++boh)
          loadRow(srcVec[boh], ic, ih + boh, iw);

        // Iterate over kernel height
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh)
        {
          // Load next input row into ring buffer
          loadRow(srcVec[(kh + blockOH - 1) % blockOH], ic, ih + (kh + blockOH - 1), iw);

          // Iterate over kernel width
          const T* weightPtr = &weight(oc, ic, kh, 0);
          
          #pragma unroll
          for (int kw = 0; kw < 3; ++kw)
          {
            // Load weights
            simd<T, blockAC * blockC> weightVec[numBlockAC];
            #pragma unroll
            for (int i = 0; i < numBlockAC; ++i)
            {
              weightVec[i].copy_from(weightPtr, vector_aligned);
              weightPtr += blockAC * blockC;
            }

            // Multiply + accumulate
            #pragma unroll
            for (int boh = 0; boh < blockOH; ++boh)
            {
              #pragma unroll
              for (int i = 0; i < numBlockAC; ++i)
              {
                accumVec[boh][i] =
                  dpas<argument_type::FP16, argument_type::FP16, AT, dpasDepth, dpasRepeat>(
                    accumVec[boh][i],
                    weightVec[i].template bit_cast_view<int>().read(),
                    srcVec[(kh + boh) % blockOH].template select<blockOW * blockC, 1>(kw * blockC).template bit_cast_view<int>().read());
              }
            }
          }
        }
      }

      // Load bias
      const auto biasVec = block_load<T, blockC>(&bias(oc), vector_aligned);
      
      #pragma unroll
      for (int boh = 0; boh < blockOH; ++boh)
      {
        if (oh + boh >= dst.H)
          break;

        // Shuffle and convert accumulators
        simd<T, blockOW * blockC> dstVec;
        auto dstMat = dstVec.template bit_cast_view<T, blockOW, blockC>();
        #pragma unroll
        for (int i = 0; i < numBlockAC; ++i)
          dstMat.template select<blockOW, 1, blockAC, 1>(0, i * blockAC) = accumVec[boh][i];

        // Add bias
        dstVec += biasVec.template replicate<blockOW>();

        // Apply ReLU
        dstVec = max(dstVec, simd<T, blockOW * blockC>(0));

        // Store output row
        T* dstPtr = &dst(oc, oh + boh, ow);
        if (ow + blockOW <= dst.W)
        {
          dstVec.copy_to(dstPtr, overaligned<32>);
        }
        else
        {
          #pragma unroll
          for (int bow = 0; bow < blockOW; ++bow)
          {
            if (ow + bow < dst.W)
              block_store<T, blockC>(dstPtr, dstVec.template select<blockC, 1>(bow * blockC));
            dstPtr += blockC;
          }
        }
      }
    }

    OIDN_INLINE void loadRow(simd<T, blockIW * blockC>& srcVec, int ic, int ih, int iw) const
    {
      if (ih < 0 || ih >= src.H)
      {
        srcVec = 0;
        return;
      }

      const T* srcPtr = &src(ic, ih, iw);

      if (iw >= 0 && iw + blockIW <= src.W)
      {
        srcVec.copy_from(srcPtr, overaligned<32>);
      }
      else
      {
        srcVec = 0;
        #pragma unroll
        for (int biw = 0; biw < blockIW; ++biw)
        {
          if (iw + biw >= 0 && iw + biw < src.W)
            srcVec.template select<blockC, 1>(biw * blockC) = block_load<T, blockC>(srcPtr, vector_aligned);
          srcPtr += blockC;
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

    using Kernel = SYCLConvDPASKernel<half, TensorLayout::Chw16c, TensorLayout::OIhw2o8i8o2i>;

    Kernel kernel;
    kernel.src    = *src;
    kernel.weight = *weight;
    kernel.bias   = *bias;
    kernel.dst    = *dst;

    WorkDim<3> globalSize = {dst->getCB(),
                             ceil_div(dst->getH(), Kernel::blockOH),
                             ceil_div(dst->getW(), Kernel::blockOW)};

    // FIXME: need to round up WB dimension to multiple of 2 due to DPAS bug
    if (globalSize[0] % 2 != 0 && globalSize[1] % 2 != 0 && globalSize[2] % 2 != 0)
      globalSize[2]++;

    WorkDim<3> localSize = {globalSize[0], 1, 1};
    int totalSize = globalSize[0];

    while (totalSize % 2 != 0 || totalSize * 2 <= 8)
    {
      const int i = (localSize[1] * Kernel::blockOH < localSize[2] * Kernel::blockOW) ? 1 : 2;
      if (globalSize[i] % (localSize[i]*2) == 0)
      {
        localSize[i] *= 2;
        totalSize *= 2;
      }
      else if (globalSize[3-i] % (localSize[3-i]*2) == 0)
      {
        localSize[3-i] *= 2;
        totalSize *= 2;
      }
      else
        break;
    }

    device->runESIMDKernelAsync(globalSize / localSize, localSize, kernel);
  }

} // namespace oidn