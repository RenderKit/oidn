// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// This file may be included multiple times in the same cpp with different macros defined,
// so we have *no* 'pragma once' here

#include "sycl_conv.h"
#include "sycl_common.h"

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_ARCH_XELP)
namespace xelp {
#elif defined(OIDN_ARCH_XEHPG)
namespace xehpg {
#elif defined(OIDN_ARCH_XEHPC)
namespace xehpc {
#elif defined(OIDN_ARCH_XE2)
namespace xe2 {
#endif

  template<typename SrcDstT, typename WeightT, TensorLayout srcDstLayout, TensorLayout weightLayout,
           PostOp postOp>
  struct SYCLConvKernel
  {
    static constexpr int KW = 3;        // kernel width
    static constexpr int KH = 3;        // kernel height
    static constexpr int PW = (KW-1)/2; // padding width on each side
    static constexpr int PH = (KH-1)/2; // padding height on each side

    static constexpr int dpasDepth  = 8; // DPAS depth
    static constexpr int dpasRepeat = 8; // DPAS repeat count

    static constexpr int blockC = TensorByteOffset<SrcDstT, srcDstLayout>::blockC; // block input/output channels

  #if defined(OIDN_ARCH_XEHPG)
    using MatmulT = SrcDstT;
    static constexpr int blockAC = 8;                   // block accumulator channels (exec width)
    static constexpr int numBlockAC = blockC / blockAC; // number of accumulator channel blocks
    static constexpr int blockOH = (postOp == PostOp::Pool) ? 6 : 5; // block output height
  #elif defined(OIDN_ARCH_XEHPC)
    using MatmulT = SrcDstT;
    static constexpr int blockOH = 4; // block output height
  #elif defined(OIDN_ARCH_XE2)
    using MatmulT = SrcDstT;
    static constexpr int blockOH = 6; // block output height
  #else
    using MatmulT = float; // no DPAS -> use FP32 FMAs
    static constexpr int blockOH = 2; // block output height
  #endif

    static constexpr int blockOW = dpasRepeat;       // block output width
    static constexpr int blockIW = blockOW + KW - 1; // block input width

    TensorAccessor3D<SrcDstT, srcDstLayout> src;
    TensorAccessor4D<WeightT, weightLayout> weight;
    TensorAccessor1D<SrcDstT> bias;
    TensorAccessor3D<SrcDstT, srcDstLayout> dst;
    //Activation activation;

    oidn_inline void operator ()(const WorkGroupItem<3>& it) const SYCL_ESIMD_FUNCTION
    {
    #if defined(OIDN_ARCH_XEHPG)
      // FP32 accumulator rows
      simd<float, blockOW * blockAC> accumRows[blockOH][numBlockAC] = {}; // = 0
    #else
      // FP32 accumulator rows
      simd<float, blockOW * blockC> accumRows[blockOH] = {}; // = 0
    #endif

      const int oc = it.getLocalID<0>()  * blockC;
      const int oh = it.getGlobalID<1>() * blockOH;
      const int ow = it.getGlobalID<2>() * blockOW;

      // Iterate over input channel blocks
      for (int ic = 0; ic < src.C; ic += blockC)
      {
        const int ih = oh - PH;
        const int iw = ow - PW;

        // Load input rows into a ring buffer
        simd<MatmulT, blockIW * blockC> inRows[blockOH];

        #pragma unroll
        for (int boh = 0; boh < blockOH - 1; ++boh)
          loadRow(inRows[boh], ic, ih + boh, iw);

        // Iterate over kernel height
        #pragma unroll
        for (int kh = 0; kh < KH; ++kh)
        {
          // Load next input row into ring buffer
          loadRow(inRows[(kh + blockOH - 1) % blockOH], ic, ih + (kh + blockOH - 1), iw);

          // Get pointer to weights for kernel row
          const WeightT* weightPtr = &weight(oc, ic, kh, 0);

          // Iterate over kernel width
          #pragma unroll
          for (int kw = 0; kw < KW; ++kw)
          {
            // Load weight matrix for kernel tap
          #if defined(OIDN_ARCH_XEHPG)
            simd<MatmulT, blockAC * blockC> weightMat[numBlockAC];
            #pragma unroll
            for (int i = 0; i < numBlockAC; ++i)
            {
              weightMat[i] = loadBlock<WeightT, blockAC * blockC>(weightPtr);
              weightPtr += blockAC * blockC;
            }
          #else
            simd<MatmulT, blockC * blockC> weightMat = loadLargeBlock<WeightT, blockC * blockC>(weightPtr);
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
          #elif defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
            #pragma unroll
            for (int boh = 0; boh < blockOH; ++boh)
            {
              accumRows[boh] = xmx::dpas<dpasDepth, dpasRepeat, float>(
                accumRows[boh],
                weightMat,
                inRows[(kh + boh) % blockOH].template select<blockOW * blockC, 1>(kw * blockC).read());
            }
          #else
            #pragma unroll
            for (int boh = 0; boh < blockOH; ++boh)
            {
              #pragma unroll
              for (int bow = 0; bow < blockOW; ++bow)
              {
                #pragma unroll
                for (int i = 0; i < blockC; ++i)
                {
                  accumRows[boh].template select<blockC, 1>(bow * blockC) +=
                    inRows[(kh + boh) % blockOH].template replicate_w<blockC, 1>((kw + bow) * blockC + i) *
                    weightMat.template select<blockC, 1>(i * blockC);
                }
              }
            }
          #endif
          }
        }
      }

    #if defined(OIDN_ARCH_XEHPG)
      // Shuffle and down-convert accumulator rows to output rows
      simd<SrcDstT, blockOW * blockC> outRows[blockOH];

      #pragma unroll
      for (int boh = 0; boh < blockOH; ++boh)
      {
        auto outRowView = outRows[boh].template bit_cast_view<SrcDstT, blockOW, blockC>();
        #pragma unroll
        for (int i = 0; i < numBlockAC; ++i)
          outRowView.template select<blockOW, 1, blockAC, 1>(0, i * blockAC) = accumRows[boh][i];
      }
    #else
      // Down-convert accumulator rows to output rows
      simd<SrcDstT, blockOW * blockC> outRows[blockOH];

      #pragma unroll
      for (int boh = 0; boh < blockOH; ++boh)
        outRows[boh] = accumRows[boh];
    #endif

      // Load bias vector
      const auto biasVec = loadBlock<SrcDstT, blockC>(&bias(oc));

      // Add bias
      #pragma unroll
      for (int boh = 0; boh < blockOH; ++boh)
        outRows[boh] += biasVec.template replicate<blockOW>();

      // Apply activation
      //if (activation == Activation::ReLU)
      {
        #pragma unroll
        for (int boh = 0; boh < blockOH; ++boh)
          outRows[boh] = max(outRows[boh], simd<SrcDstT, blockOW * blockC>(0));
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
          simd<SrcDstT, blockOW * blockC * 2> upRow1x2;

          #pragma unroll
          for (int bow = 0; bow < blockOW; ++bow)
          {
            upRow1x2.template select<blockC * 2, 1>(bow * blockC * 2) =
              outRows[boh].template replicate_w<2, blockC>(bow * blockC);
          }

          // Store upsampled rows
          storeRow<2>(upRow1x2, oc, (oh + boh) * 2,     ow * 2);
          storeRow<2>(upRow1x2, oc, (oh + boh) * 2 + 1, ow * 2);
        }
      }
    }

    // Loads a row from the src tensor
    template<int N>
    oidn_inline void loadRow(simd<MatmulT, N>& row, int ic, int ih, int iw) const
    {
      static_assert(N % blockC == 0, "non-integer width");
      constexpr int W = N / blockC;

      if (ih < 0 || ih >= src.H)
      {
        row = 0;
        return;
      }

      if (iw >= 0 && iw + W <= src.W)
      {
        // Fast path: load the entire row
        const SrcDstT* srcPtr = &src(ic, ih, iw);
        row = loadLargeBlock<SrcDstT, N>(srcPtr);
      }
      else
      {
        // Slow path: load the in-bounds pixels of the row
        const simd<int, W> iwVec(iw, 1); // iw, iw+1, iw+2, ...
        simd_mask<W> predVec = (iwVec >= 0) & (iwVec < src.W);
        simd<uint32_t,  W> srcOffsetVec = src.getByteOffset(ic, ih, 0) +
                                          iwVec * src.getByteOffset.wByteStride;
        simd<uintptr_t, W> srcAddrVec   = reinterpret_cast<uintptr_t>(src.ptr) + srcOffsetVec;

        #pragma unroll
        for (int w = 0; w < W; ++w)
        {
          const SrcDstT* srcPtr = reinterpret_cast<const SrcDstT*>(uintptr_t(srcAddrVec[w]));
          row.template select<blockC, 1>(w * blockC) =
            loadBlock<SrcDstT, blockC>(srcPtr, predVec.template select<1, 1>(w));
        }
      }
    }

    // Stores a row to the dst tensor
    // Pixels can be stored in chunks of K to improve performance
    template<int K = 1, int N>
    oidn_inline void storeRow(simd<SrcDstT, N>& row, int oc, int oh, int ow) const
    {
      static_assert(N % blockC == 0, "non-integer width");
      constexpr int W = N / blockC;
      static_assert(W % K == 0, "non-integer chunks");

      //if (oh >= dst.H)
      //  return;

      if (ow + W <= dst.W)
      {
        // Fast path: store the entire row
        SrcDstT* dstPtr = &dst(oc, oh, ow);
        storeLargeBlock(dstPtr, row);
      }
      else
      {
        // Slow path: store the in-bounds pixels of the row
        constexpr int numChunks = W / K;
        constexpr int chunkSize = blockC * K;

        const simd<int, numChunks> owVec(ow, K); // ow, ow+K, ow+2*K, ...
        simd_mask<numChunks> predVec = owVec < dst.W;
        simd<uint32_t,  numChunks> dstOffsetVec = dst.getByteOffset(oc, oh, 0) +
                                                  owVec * dst.getByteOffset.wByteStride;
        simd<uintptr_t, numChunks> dstAddrVec   = reinterpret_cast<uintptr_t>(dst.ptr) + dstOffsetVec;

        #pragma unroll
        for (int i = 0; i < numChunks; ++i)
        {
          SrcDstT* dstPtr = reinterpret_cast<SrcDstT*>(uintptr_t(dstAddrVec[i]));
          storeBlock(dstPtr, row.template select<chunkSize, 1>(i * chunkSize).read(),
                     predVec.template select<1, 1>(i));
        }
      }
    }
  };

  template<typename SrcDstT>
  class SYCLConv : public Conv
  {
  public:
    SYCLConv(SYCLEngine* engine, const ConvDesc& desc)
      : Conv(desc),
        engine(engine)
    {
      if (srcDesc.layout != srcDstLayout || srcDesc.dataType != DataTypeOf<SrcDstT>::value)
        throw std::invalid_argument("unsupported convolution source layout/data type");
      if (weightDesc.getW() != 3 || weightDesc.getH() != 3)
        throw std::invalid_argument("unsupported convolution kernel size");
      if (weightDesc.layout != weightLayout || weightDesc.dataType != DataTypeOf<WeightT>::value)
        throw std::invalid_argument("unsupported convolution weight layout/data type");
      if (biasDesc.layout != TensorLayout::x || biasDesc.dataType != DataTypeOf<SrcDstT>::value)
        throw std::invalid_argument("unsupported convolution bias layout/data type");

      if (desc.activation != Activation::ReLU)
        throw std::invalid_argument("unsupported convolution activation");
    }

    Engine* getEngine() const override { return engine; }

    void submitKernels(const Ref<CancellationToken>& ct) override
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
    static constexpr TensorLayout srcDstLayout = TensorLayout::Chw16c;

  #if defined(OIDN_ARCH_XEHPG)
    using WeightT = SrcDstT;
    static constexpr TensorLayout weightLayout = TensorLayout::OIhw2o8i8o2i;
  #elif defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
    using WeightT = SrcDstT;
    static constexpr TensorLayout weightLayout = TensorLayout::OIhw8i16o2i;
  #else
    using WeightT = float;
    static constexpr TensorLayout weightLayout = TensorLayout::OIhw16i16o;
  #endif

    template<PostOp kernelPostOp>
    void submitImpl()
    {
      using Kernel = SYCLConvKernel<SrcDstT, WeightT, srcDstLayout, weightLayout, kernelPostOp>;

      Kernel kernel;
      kernel.src    = *src;
      kernel.weight = *weight;
      kernel.bias   = *bias;
      kernel.dst    = *dst;
      //kernel.activation = activation;

      WorkDim<3> globalSize = {dst->getPaddedC() / Kernel::blockC,
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
    #if defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
      const int maxGroupSize = (engine->getArch() == SYCLArch::XeHPC) ? 32 : 8;
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

    #if defined(OIDN_ARCH_XEHPG)
      engine->submitESIMDKernelWithLargeGRF(globalSize / groupSize, groupSize, kernel);
    #else
      engine->submitESIMDKernel(globalSize / groupSize, groupSize, kernel);
    #endif
    }

    SYCLEngine* engine;
  };

  Ref<Conv> newSYCLConv(SYCLEngine* engine, const ConvDesc& desc)
  {
    return makeRef<SYCLConv<half>>(engine, desc);
  }

} // namespace arch

OIDN_NAMESPACE_END