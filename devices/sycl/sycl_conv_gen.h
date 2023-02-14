// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sycl_conv.h"
#include "sycl_common.h"
#include <sycl/ext/intel/experimental/kernel_properties.hpp>

OIDN_NAMESPACE_BEGIN
#if defined(OIDN_ARCH_XEHPG)
namespace xehpg {
#elif defined(OIDN_ARCH_XEHPC)
namespace xehpc {
#else
namespace gen9 {
#endif

  template<typename T, TensorLayout tensorLayout, TensorLayout weightLayout, PostOp postOp>
  struct SYCLConvKernel
  {
    static constexpr int dpasDepth  = 8; // DPAS depth
    static constexpr int dpasRepeat = 8; // DPAS repeat count

    static constexpr int blockC = TensorAccessor3D<T, tensorLayout>::blockC; // block input/output channels

  #if defined(OIDN_ARCH_XEHPG)
    static constexpr int blockAC = 8;                   // block accumulator channels (exec width)
    static constexpr int numBlockAC = blockC / blockAC; // number of accumulator channel blocks
    
    static constexpr int blockOH = (postOp == PostOp::Pool) ? 6 : 5; // block output height
  #elif defined(OIDN_ARCH_XEHPC)
    static constexpr int blockOH = 4; // block output height
  #else
    static constexpr int blockOH = 2; // block output height
  #endif

    static constexpr int blockOW = dpasRepeat;      // block output width
    static constexpr int blockIW = blockOW + 3 - 1; // block input width
    
    TensorAccessor3D<T, tensorLayout> src;
    TensorAccessor4D<T, weightLayout> weight;
    TensorAccessor1D<T> bias;
    TensorAccessor3D<T, tensorLayout> dst;

    OIDN_INLINE void operator ()(const WorkGroupItem<3>& it) const SYCL_ESIMD_FUNCTION
    {
    #if defined(OIDN_ARCH_XEHPG)
      syclx::set_kernel_properties(syclx::kernel_properties::use_large_grf);

      // Accumulator rows
      simd<float, blockOW * blockAC> accumRows[blockOH][numBlockAC] = {}; // = 0
    #else
      // Output rows
      simd<T, blockOW * blockC> outRows[blockOH] = {}; // = 0
    #endif

      const int oc = it.getLocalId<0>()  * blockC;
      const int oh = it.getGlobalId<1>() * blockOH;
      const int ow = it.getGlobalId<2>() * blockOW;

      // Iterate over input channel blocks
      for (int ic = 0; ic < src.C; ic += blockC)
      {
        const int ih = oh - 1;
        const int iw = ow - 1;

        // Load input rows into a ring buffer
        simd<T, blockIW * blockC> inRows[blockOH];

        #pragma unroll
        for (int boh = 0; boh < blockOH - 1; ++boh)
          loadRow(inRows[boh], ic, ih + boh, iw);

        // Iterate over kernel height
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh)
        {
          // Load next input row into ring buffer
          loadRow(inRows[(kh + blockOH - 1) % blockOH], ic, ih + (kh + blockOH - 1), iw);

          // Get pointer to weights for kernel row
          const T* weightPtr = &weight(oc, ic, kh, 0);

          // Iterate over kernel width
          #pragma unroll
          for (int kw = 0; kw < 3; ++kw)
          {
            // Load weight matrix for kernel tap
          #if defined(OIDN_ARCH_XEHPG)
            simd<T, blockAC * blockC> weightMat[numBlockAC];
            #pragma unroll
            for (int i = 0; i < numBlockAC; ++i)
            {
              weightMat[i] = loadBlock<T, blockAC * blockC>(weightPtr);
              weightPtr += blockAC * blockC;
            }
          #else
            simd<T, blockC * blockC> weightMat;
            loadLargeBlock<T, blockC * blockC>(weightPtr, weightMat);
            weightPtr += blockC * blockC;
          #endif

            // Multiply + accumulate rows
          #if defined(OIDN_ARCH_XEHPG)
            #pragma unroll
            for (int boh = 0; boh < blockOH; ++boh)
            {
              #pragma unroll
              for (int i = 0; i < numBlockAC; ++i)
              {
                accumRows[boh][i] = xmx::dpas<dpasDepth, dpasRepeat, float>(
                  accumRows[boh][i],
                  weightMat[i],
                  inRows[(kh + boh) % blockOH].template select<blockOW * blockC, 1>(kw * blockC).read());
              }
            }
          #elif defined(OIDN_ARCH_XEHPC)
            #pragma unroll
            for (int boh = 0; boh < blockOH; ++boh)
            {
              outRows[boh] = xmx::dpas<dpasDepth, dpasRepeat, T>(
                outRows[boh],
                weightMat,
                inRows[(kh + boh) % blockOH].template select<blockOW * blockC, 1>(kw * blockC).read());
            }
          #else
            #pragma unroll
            for (int i = 0; i < blockC; ++i)
            {
              #pragma unroll
              for (int boh = 0; boh < blockOH; ++boh)
              {
                #pragma unroll
                for (int bow = 0; bow < blockOW; ++bow)
                  outRows[boh].template select<blockC, 1>(bow * blockC) +=
                    inRows[(kh + boh) % blockOH].template replicate_w<blockC, 1>((kw + bow) * blockC + i) *
                    weightMat.template select<blockC, 1>(i * blockC);
              }
            }
          #endif
          }
        }
      }

    #if defined(OIDN_ARCH_XEHPG)
      // Shuffle and convert accumulator rows to output rows
      simd<T, blockOW * blockC> outRows[blockOH];
      
      #pragma unroll
      for (int boh = 0; boh < blockOH; ++boh)
      {
        auto outRowView = outRows[boh].template bit_cast_view<T, blockOW, blockC>();
        #pragma unroll
        for (int i = 0; i < numBlockAC; ++i)
          outRowView.template select<blockOW, 1, blockAC, 1>(0, i * blockAC) = accumRows[boh][i];
      }
    #endif

      // Load bias vector
      const auto biasVec = loadBlock<T, blockC>(&bias(oc));

      #pragma unroll
      for (int boh = 0; boh < blockOH; ++boh)
      {
        // Add bias
        outRows[boh] += biasVec.template replicate<blockOW>();

        // Apply ReLU
        outRows[boh] = max(outRows[boh], simd<T, blockOW * blockC>(0));
      }
      
      // Store output rows
      if constexpr (postOp == PostOp::None)
      {
        #pragma unroll
        for (int boh = 0; boh < blockOH; ++boh)
        {
          if (oh + boh >= dst.H)
            break;

          // Store output row
          storeRow(outRows[boh], oc, oh + boh, ow);
        }
      }
      else if constexpr (postOp == PostOp::Pool)
      {
        #pragma unroll
        for (int boh = 0; boh < blockOH; boh += 2)
        {
          if (oh + boh >= src.H) // src.H = output height without pooling
            break;

          // Pool output rows
          auto poolRow2x1 = max(outRows[boh], outRows[boh + 1]);
          auto poolRow2x2 = max(poolRow2x1.template replicate_vs_w<blockOW / 2, blockC * 2, blockC>(0),
                                poolRow2x1.template replicate_vs_w<blockOW / 2, blockC * 2, blockC>(blockC));
          
          // Store pooled row
          storeRow(poolRow2x2, oc, (oh + boh) / 2, ow / 2);
        }
      }
      else if constexpr (postOp == PostOp::Upsample)
      {
        #pragma unroll
        for (int boh = 0; boh < blockOH; ++boh)
        {
          if (oh + boh >= src.H) // src.H = output height without upsampling
            break;

          // Upsample output row
          simd<T, blockOW * blockC * 2> upRow1x2;

          #pragma unroll
          for (int bow = 0; bow < blockOW; ++bow)
            upRow1x2.template select<blockC * 2, 1>(bow * blockC * 2) = outRows[boh].template replicate_w<2, blockC>(bow * blockC);

          // Store upsampled rows
          storeRow<2>(upRow1x2, oc, (oh + boh) * 2,     ow * 2);
          storeRow<2>(upRow1x2, oc, (oh + boh) * 2 + 1, ow * 2);
        }
      }
    }

    // Loads a row from the src tensor
    template<int N>
    OIDN_INLINE void loadRow(simd<T, N>& row, int ic, int ih, int iw) const
    {
      static_assert(N % blockC == 0, "non-integer width");
      constexpr int W = N / blockC;

      if (ih < 0 || ih >= src.H)
      {
        row = 0;
        return;
      }

      const T* srcPtr = &src(ic, ih, iw);

      if (iw >= 0 && iw + W <= src.W)
      {
        // Fast path: load the entire row
        loadLargeBlock(srcPtr, row);
      }
      else
      {
        // Slow path: load the in-bounds pixels of the row
        const simd<int, W> wVec(0, 1); // 0, 1, 2, ...
        simd_mask<W> predVec = (wVec >= -iw) & (wVec < src.W - iw);

        #pragma unroll
        for (int w = 0; w < W; ++w)
        {
          row.template select<blockC, 1>(w * blockC) = loadBlock<T, blockC>(srcPtr, predVec.template select<1, 1>(w));
          srcPtr += blockC;
        }
      }
    }

    // Stores a row to the dst tensor
    // Pixels can be stored in chunks of K to improve performance
    template<int K = 1, int N>
    OIDN_INLINE void storeRow(simd<T, N>& row, int oc, int oh, int ow) const
    {
      static_assert(N % blockC == 0, "non-integer width");
      constexpr int W = N / blockC;
      static_assert(W % K == 0, "non-integer chunks");

      //if (oh >= dst.H)
      //  return;

      T* dstPtr = &dst(oc, oh, ow);

      if (ow + W <= dst.W)
      {
        // Fast path: store the entire row
        storeLargeBlock(dstPtr, row);
      }
      else
      {
        // Slow path: store the in-bounds pixels of the row
        constexpr int numChunks = W / K;
        constexpr int chunkSize = blockC * K;
        const simd<int, numChunks> wVec(0, K); // 0, 1*K, 2*K, ...
        simd_mask<numChunks> predVec = wVec < dst.W - ow;

        #pragma unroll
        for (int i = 0; i < numChunks; ++i)
        {
          storeBlock(dstPtr, row.template select<chunkSize, 1>(i * chunkSize).read(), predVec.template select<1, 1>(i));
          dstPtr += chunkSize;
        }
      }
    }
  };

  class SYCLConv : public Conv
  {
  public:
    SYCLConv(const Ref<SYCLEngine>& engine, const ConvDesc& desc)
      : Conv(desc),
        engine(engine)
    {
      if (srcDesc.layout != tensorLayout || srcDesc.dataType != DataType::Float16)
        throw std::invalid_argument("unsupported convolution source layout/data type");
      if (weightDesc.layout != weightLayout || weightDesc.dataType != DataType::Float16)
        throw std::invalid_argument("unsupported convolution weight layout/data type");
      if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataType::Float16)
        throw std::invalid_argument("unsupported convolution bias layout/data type");
    }

    void submit() override
    {
      if (!src || !weight || !bias || !dst)
        throw std::logic_error("convolution argument not set");

      switch (postOp)
      {
      case PostOp::None:
        submitImpl<PostOp::None>();
        break;
      case PostOp::Pool:
        submitImpl<PostOp::Pool>();
        break;
      case PostOp::Upsample:
        submitImpl<PostOp::Upsample>();
        break;
      }
    }

  private:
    static constexpr TensorLayout tensorLayout = TensorLayout::Chw16c;

  #if defined(OIDN_ARCH_XEHPG)
    static constexpr TensorLayout weightLayout = TensorLayout::OIhw2o8i8o2i;
  #elif defined(OIDN_ARCH_XEHPC)
    static constexpr TensorLayout weightLayout = TensorLayout::OIhw8i16o2i;
  #else
    static constexpr TensorLayout weightLayout = TensorLayout::OIhw16i16o;
  #endif

    template<PostOp kernelPostOp>
    void submitImpl()
    {
      using Kernel = SYCLConvKernel<half, tensorLayout, weightLayout, kernelPostOp>;

      Kernel kernel;
      kernel.src    = *src;
      kernel.weight = *weight;
      kernel.bias   = *bias;
      kernel.dst    = *dst;

      WorkDim<3> globalSize = {dst->getC() / Kernel::blockC,
                               ceil_div(src->getH(), Kernel::blockOH),
                               ceil_div(src->getW(), Kernel::blockOW)};

      WorkDim<3> groupSize = {globalSize[0], 1, 1};

    #if defined(OIDN_ARCH_XEHPG)
      // Workaround for DPAS + EU fusion bug: make sure to have even number of threads per group
      if (globalSize[0] % 2 != 0 && globalSize[1] % 2 != 0 && globalSize[2] % 2 != 0)
      {
        // We can safely round up one of the spatial dimensions thanks to bounds checking in the kernel
        globalSize[2]++;
        groupSize[2]++;
      }
    #endif

      // Compute the final work-group size
    #if defined(OIDN_ARCH_XEHPC)
      const int maxGroupSize = 32;
    #else
      const int maxGroupSize = 16;
    #endif
    
      for (; ;)
      {
        bool updated = false;

        // Try to increase one of the spatial dimensions (1 or 2), smallest first
        int dim = (groupSize[1] * Kernel::blockOH < groupSize[2] * Kernel::blockOW) ? 1 : 2; 
        for (int i = 0; i < 2 && !updated; ++i, dim = 3-dim)
        {
          const int maxDiv = maxGroupSize / (groupSize[0] * groupSize[3-dim]);
          for (int div = groupSize[dim] + 1; div <= maxDiv && !updated; ++div)
          {
            if (globalSize[dim] % div == 0
              #if defined(OIDN_ARCH_XEHPG)
                && (groupSize[0] * groupSize[3-dim] * div) % 2 == 0 // must have even number of threads
              #endif
               )
            {
              groupSize[dim] = div;
              updated = true;
            }
          }
        }
        
        if (!updated)
          break;
      }

      engine->submitESIMDKernel(globalSize / groupSize, groupSize, kernel);
    }

    Ref<SYCLEngine> engine;
  };

  std::shared_ptr<Conv> newConv(const Ref<SYCLEngine>& engine, const ConvDesc& desc)
  {
    return std::make_shared<SYCLConv>(engine, desc);
  }

} // namespace arch
OIDN_NAMESPACE_END