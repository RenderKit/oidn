// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// FIXME: ESIMD compile error on Windows
#if defined(_WIN32)
  typedef unsigned int uint;
#endif

#include <sycl/ext/intel/experimental/esimd/math.hpp>
#include <sycl/ext/intel/experimental/esimd/memory.hpp>
#include "sycl_conv_dpas.h"

// FIXME: add to ESIMD
__SYCL_INLINE_NAMESPACE(cl) {
namespace __ESIMD_ENS {

template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__ESIMD_API __ESIMD_NS::simd<T, NElts> lsc_block_load(const T *p, __ESIMD_NS::simd_mask<1> pred/* = 1*/) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::load, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  static_assert(_DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
                "Transposed load is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  return __esimd_lsc_load_stateless<T, L1H, L3H, _AddressScale, _ImmOffset, _DS,
                                    _VS, _Transposed, N>(pred.data(),
                                                         addrs.data());
}

template <typename T, uint8_t NElts = 1,
          lsc_data_size DS = lsc_data_size::default_size,
          cache_hint L1H = cache_hint::none, cache_hint L3H = cache_hint::none>
__ESIMD_API void lsc_block_store(T *p, __ESIMD_NS::simd<T, NElts> vals, __ESIMD_NS::simd_mask<1> pred/* = 1*/) {
  detail::check_lsc_vector_size<NElts>();
  detail::check_lsc_data_size<T, DS>();
  detail::check_lsc_cache_hint<detail::lsc_action::store, L1H, L3H>();
  constexpr uint16_t _AddressScale = 1;
  constexpr int _ImmOffset = 0;
  constexpr lsc_data_size _DS = detail::finalize_data_size<T, DS>();
  static_assert(_DS == lsc_data_size::u32 || _DS == lsc_data_size::u64,
                "Transposed store is supported only for data size u32 or u64");
  constexpr detail::lsc_vector_size _VS = detail::to_lsc_vector_size<NElts>();
  constexpr detail::lsc_data_order _Transposed =
      detail::lsc_data_order::transpose;
  constexpr int N = 1;
  __ESIMD_NS::simd<uintptr_t, N> addrs = reinterpret_cast<uintptr_t>(p);
  __esimd_lsc_store_stateless<T, L1H, L3H, _AddressScale, _ImmOffset, _DS, _VS,
                              _Transposed, N>(pred.data(), addrs.data(),
                                              vals.data());
}

} // namespace __ESIMD_ENS
} // __SYCL_INLINE_NAMESPACE(cl)

namespace oidn {

  using namespace esimd;
  using namespace sycl::ext::intel::experimental::esimd;

  template<typename T, int N>
  OIDN_INLINE simd<T, N> loadBlock(const T* ptr)
  {
    //return block_load<T, N>(ptr, vector_aligned);
    
    static_assert((sizeof(T) * N) % sizeof(int) == 0, "unsupported block size");
    auto blk = lsc_block_load<int, (sizeof(T) * N) / sizeof(int)>((const int*)ptr);
    return blk.template bit_cast_view<T>();
  }

  template<typename T, int N>
  OIDN_INLINE simd<T, N> loadBlock(const T* ptr, simd_mask<1> pred)
  {
    //return block_load<T, N>(ptr, vector_aligned);
    
    static_assert((sizeof(T) * N) % sizeof(int) == 0, "unsupported block size");
    auto blk = lsc_block_load<int, (sizeof(T) * N) / sizeof(int)>((const int*)ptr, pred);
    auto res = simd<T, N>(0);
    res.merge(blk.template bit_cast_view<T>(), simd_mask<N>(pred[0]));
    return res;
  }

  template<typename T, int N>
  OIDN_INLINE void storeBlock(T* ptr, simd<T, N> blk, simd_mask<1> pred = 1)
  {
    //block_store(ptr, blk);

    static_assert((sizeof(T) * N) % sizeof(int) == 0, "unsupported block size");
    lsc_block_store<int, (sizeof(T) * N) / sizeof(int)>((int*)ptr, blk.template bit_cast_view<int>(), pred);
  }

  template<typename T, int N>
  OIDN_INLINE void loadLargeBlock(const T* ptr, simd<T, N>& blk)
  {
    //blk.copy_from(ptr, overaligned<32>);

    constexpr int chunkSize = 256 / sizeof(T);
    constexpr int numChunks = N / chunkSize;
    constexpr int remSize   = N % chunkSize;

    #pragma unroll
    for (int i = 0; i < numChunks; ++i)
      blk.template select<chunkSize, 1>(i * chunkSize) = loadBlock<T, chunkSize>(ptr + i * chunkSize);

    if constexpr (remSize != 0)
      blk.template select<remSize, 1>(numChunks * chunkSize) = loadBlock<T, remSize>(ptr + numChunks * chunkSize);
  }

  template<typename T, int N>
  OIDN_INLINE void storeLargeBlock(T* ptr, simd<T, N>& blk)
  {
    //blk.copy_to(ptr, overaligned<32>);

    constexpr int chunkSize = 256 / sizeof(T);
    constexpr int numChunks = N / chunkSize;
    constexpr int remSize   = N % chunkSize;

    #pragma unroll
    for (int i = 0; i < numChunks; ++i)
      storeBlock<T, chunkSize>(ptr + i * chunkSize, blk.template select<chunkSize, 1>(i * chunkSize));

    if constexpr (remSize != 0)
      storeBlock<T, remSize>(ptr + numChunks * chunkSize, blk.template select<remSize, 1>(numChunks * chunkSize));
  }

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
              weightVec[i] = loadBlock<T, blockAC * blockC>(weightPtr);
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
      const auto biasVec = loadBlock<T, blockC>(&bias(oc));
      
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
          storeLargeBlock(dstPtr, dstVec);
        }
        else
        {
          const simd<int, blockOW> owVec(ow, 1);
          simd_mask<blockOW> predVec = owVec < dst.W;

          #pragma unroll
          for (int bow = 0; bow < blockOW; ++bow)
          {
            storeBlock(dstPtr, dstVec.template select<blockC, 1>(bow * blockC).read(), predVec.select<1, 1>(bow));
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
        loadLargeBlock(srcPtr, srcVec);
      }
      else
      {
        const simd<int, blockIW> iwVec(iw, 1);
        simd_mask<blockIW> predVec = (iwVec >= 0) & (iwVec < src.W);

        #pragma unroll
        for (int biw = 0; biw < blockIW; ++biw)
        {
          srcVec.template select<blockC, 1>(biw * blockC) = loadBlock<T, blockC>(srcPtr, predVec.select<1, 1>(biw));
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