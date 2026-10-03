// Copyright 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

#if defined(OIDN_ARCH_XELP)
#define OIDN_ARCH_XE_NAMESPACE xelp
#elif defined(OIDN_ARCH_XEHPG)
#define OIDN_ARCH_XE_NAMESPACE xehpg
#elif defined(OIDN_ARCH_XEHPC)
#define OIDN_ARCH_XE_NAMESPACE xehpc
#elif defined(OIDN_ARCH_XE2)
#define OIDN_ARCH_XE_NAMESPACE xe2
#endif

#define OIDN_ARCH_XE_NAMESPACE_BEGIN namespace OIDN_ARCH_XE_NAMESPACE {
#define OIDN_ARCH_XE_NAMESPACE_END }

OIDN_NAMESPACE_BEGIN

#if defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
  constexpr int maxLSCBlockByteSize = 512;
#elif defined(OIDN_ARCH_XEHPG)
  constexpr int maxLSCBlockByteSize = 256;
#else
  constexpr int maxLSCBlockByteSize = 128;
#endif

  // Helper class for LSC block load/store
  template<typename T, int N>
  struct LSCBlockTraits
  {
    static constexpr int byteSize = sizeof(T) * N;

    // Use 64-bit data type if the vector size would be too large with 32-bit, otherwise use 32-bit
    using DT = std::conditional_t<(byteSize > 256) && (byteSize % sizeof(int64_t) == 0), int64_t, int>;
    static constexpr int DN = byteSize / sizeof(DT);

    static_assert(byteSize % sizeof(DT) == 0, "unsupported block size");
  };

#if defined(OIDN_ARCH_XEHPG) || defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)

  template<typename T, int N>
  oidn_inline simd<T, N> loadBlock(const T* ptr)
  {
    using DT = typename LSCBlockTraits<T, N>::DT;
    constexpr int DN = LSCBlockTraits<T, N>::DN;

    auto blk = lsc_block_load<DT, DN>((const DT*)ptr);
    return blk.template bit_cast_view<T>();
  }

  template<typename T, int N>
  oidn_inline simd<T, N> loadBlock(const T* ptr, simd_mask<1> pred, simd<T, N> src = 0)
  {
    using DT = typename LSCBlockTraits<T, N>::DT;
    constexpr int DN = LSCBlockTraits<T, N>::DN;

    auto blk = lsc_block_load<DT, DN>((const DT*)ptr, pred);
    src.merge(blk.template bit_cast_view<T>(), simd_mask<N>(pred[0]));
    return src;
  }

  template<typename T, int N>
  oidn_inline void storeBlock(T* ptr, simd<T, N> blk, simd_mask<1> pred = 1)
  {
    using DT = typename LSCBlockTraits<T, N>::DT;
    constexpr int DN = LSCBlockTraits<T, N>::DN;

    lsc_block_store<DT, DN>((DT*)ptr, blk.template bit_cast_view<DT>(), pred);
  }

#else

  template<typename T, int N>
  oidn_inline simd<T, N> loadBlock(const T* ptr)
  {
    return block_load<T, N>(ptr, overaligned<16>);
  }

  template<typename T, int N>
  oidn_inline simd<T, N> loadBlock(const T* ptr, simd_mask<1> pred, simd<T, N> src = 0)
  {
    if (pred)
      return block_load<T, N>(ptr, overaligned<16>);
    else
      return src;
  }

  template<typename T, int N>
  oidn_inline void storeBlock(T* ptr, simd<T, N> blk)
  {
    block_store(ptr, blk);
  }

  template<typename T, int N>
  oidn_inline void storeBlock(T* ptr, simd<T, N> blk, simd_mask<1> pred)
  {
    if (pred)
      block_store(ptr, blk);
  }

#endif

  template<typename T, int N, int blockSize = maxLSCBlockByteSize / sizeof(T), int offset = 0>
  oidn_inline void loadLargeBlock(const T* ptr, simd<T, N>& dst)
  {
    if constexpr (offset + blockSize <= N)
    {
      dst.template select<blockSize, 1>(offset) = loadBlock<T, blockSize>(ptr + offset);
      loadLargeBlock<T, N, blockSize, offset + blockSize>(ptr, dst);
    }
    else if constexpr (offset < N)
      loadLargeBlock<T, N, blockSize / 2, offset>(ptr, dst);
  }

  template<typename T, int N, int blockSize = maxLSCBlockByteSize / sizeof(T)>
  oidn_inline simd<T, N> loadLargeBlock(const T* ptr)
  {
    simd<T, N> dst;
    loadLargeBlock<T, N, blockSize>(ptr, dst);
    return dst;
  }

  template<typename T, int N, int blockSize = maxLSCBlockByteSize / sizeof(T), int offset = 0>
  oidn_inline void storeLargeBlock(T* ptr, simd<T, N>& src)
  {
    if constexpr (offset + blockSize <= N)
    {
      storeBlock<T, blockSize>(ptr + offset, src.template select<blockSize, 1>(offset));
      storeLargeBlock<T, N, blockSize, offset + blockSize>(ptr, src);
    }
    else if constexpr (offset < N)
      storeLargeBlock<T, N, blockSize / 2, offset>(ptr, src);
  }

  // -----------------------------------------------------------------------------------------------

#if defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
  // Xe-HPC/Xe2+ 2D block loads/stores address one channel plane of a tensor as a surface: elements
  // of a block outside the surface load as zero and are not stored. The restrictions on the
  // surface, and what guarantees each for the blocked CHW tensors handled here:
  //  - base address 64-byte aligned: the tensor allocation and the C plane stride are
  //    (TensorLayoutTraitsChwBc::CByteAlignment)
  //  - width a multiple of 4 bytes, pitch >= width and a multiple of 16 bytes: both are the row
  //    size, a whole number of pixels of wByteStride (a multiple of 32) bytes each
  //  - width >= 64 bytes: a row is at least 2 pixels, the filter pads its input for that
  //    (UNetFilter::minPaddedTileW)
  //  - X (in elements) a multiple of 4 bytes: a whole number of pixels
  //  - block width a multiple of 4 bytes and at most 64 bytes, block height at most 32 rows for
  //    loads and 8 rows for stores: one 64-byte block per message, the rest is checked by ESIMD

  // Loads a row from a tensor using LSC 2D block loads with implicit zero padding
  template<typename DstT, int N, typename SrcT, TensorLayout layout>
  oidn_inline void loadRow(simd<DstT, N>& row,
                           const TensorAccessor3D<SrcT, layout>& src,
                           int ic, int ih, int iw)
  {
    constexpr int blockC = TensorByteOffset<SrcT, layout>::blockC;
    constexpr int lscBlockN = 64 / sizeof(SrcT); // LSC 2D block width in elements
    static_assert(N % lscBlockN == 0, "row size must be multiple of LSC 2D block width");
    static_assert(TensorByteOffset<SrcT, layout>::wByteStride % 32 == 0,
                  "Xe-HPC/Xe2+ 2D block surfaces need pixels of a multiple of 32 bytes");

    const SrcT* surfPtr = &src(ic, 0, 0);
    const uint surfWidth  = src.getByteOffset.hByteStride - 1; // bytes - 1
    const uint surfHeight = src.H - 1;                         // rows - 1
    const uint surfPitch  = surfWidth;

    #pragma unroll
    for (int n = 0; n < N; n += lscBlockN)
    {
      row.template select<lscBlockN, 1>(n) =
        load_2d<SrcT, lscBlockN, 1>(surfPtr, surfWidth, surfHeight, surfPitch,
                                    iw * blockC + n, ih);
    }
  }

  // Loads multiple rows from a tensor using LSC 2D block loads with implicit zero padding
  template<int numRows, typename DstT, int N, int M, typename SrcT, TensorLayout layout>
  oidn_inline void loadRows(simd<DstT, N> (&rows)[M],
                            const TensorAccessor3D<SrcT, layout>& src,
                            int ic, int ih, int iw)
  {
    constexpr int blockC = TensorByteOffset<SrcT, layout>::blockC;
    constexpr int lscBlockN = 64 / sizeof(SrcT);
    static_assert(N % lscBlockN == 0, "row size must be multiple of LSC 2D block width");
    static_assert(TensorByteOffset<SrcT, layout>::wByteStride % 32 == 0,
                  "Xe-HPC/Xe2+ 2D block surfaces need pixels of a multiple of 32 bytes");

    const SrcT* surfPtr = &src(ic, 0, 0);
    const uint surfWidth  = src.getByteOffset.hByteStride - 1; // bytes - 1
    const uint surfHeight = src.H - 1;                         // rows - 1
    const uint surfPitch  = surfWidth;

    #pragma unroll
    for (int n = 0; n < N; n += lscBlockN)
    {
      auto lscBlock = load_2d<SrcT, lscBlockN, numRows>(
        surfPtr, surfWidth, surfHeight, surfPitch,
        iw * blockC + n, ih);

      #pragma unroll
      for (int r = 0; r < numRows; ++r)
      {
        rows[r].template select<lscBlockN, 1>(n) =
          lscBlock.template select<lscBlockN, 1>(r * lscBlockN);
      }
    }
  }
#else
  // Loads a row from a tensor using regular LSC block loads with explicit bounds checking and padding
  template<typename DstT, int N, typename SrcT, TensorLayout layout>
  oidn_inline void loadRow(simd<DstT, N>& row,
                           const TensorAccessor3D<SrcT, layout>& src,
                           int ic, int ih, int iw)
  {
    constexpr int blockC = TensorByteOffset<SrcT, layout>::blockC;
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
      const SrcT* srcPtr = &src(ic, ih, iw);
      row = loadLargeBlock<SrcT, N>(srcPtr);
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
        const SrcT* srcPtr = reinterpret_cast<const SrcT*>(uintptr_t(srcAddrVec[w]));
        row.template select<blockC, 1>(w * blockC) =
          loadBlock<SrcT, blockC>(srcPtr, predVec.template select<1, 1>(w));
      }
    }
  }

  template<int numRows, typename DstT, int N, int M, typename SrcT, TensorLayout layout>
  oidn_inline void loadRows(simd<DstT, N> (&rows)[M],
                            const TensorAccessor3D<SrcT, layout>& src,
                            int ic, int ih, int iw)
  {
    #pragma unroll
    for (int r = 0; r < numRows; ++r)
      loadRow(rows[r], src, ic, ih + r, iw);
  }
#endif

  // Loads a row from a virtually 2x upsampled tensor
  // alignW must indicate whether iw is even or odd
  template<int alignW, typename DstT, int N, typename SrcT, TensorLayout layout>
  oidn_inline void loadUpsampleRow(simd<DstT, N>& row,
                                   const TensorAccessor3D<SrcT, layout>& src,
                                   int ic, int ih, int iw)
  {
    constexpr int blockC = TensorByteOffset<SrcT, layout>::blockC;
    static_assert(N % blockC == 0, "non-integer width");
    constexpr int W = N / blockC;
    static_assert(W % 2 == 0, "width must be even for 2x upscale");

    ih >>= 1;
    iw >>= 1;

    constexpr int srcW = (W >> 1) + alignW; // number of source pixels to load

    simd<SrcT, srcW * blockC> srcRow;
    loadRow(srcRow, src, ic, ih, iw);

    // Upsample by copying the pixels from the source row to the destination row
    #pragma unroll
    for (int i = 0; i < W; ++i)
    {
      row.template select<blockC, 1>(i * blockC) =
        srcRow.template select<blockC, 1>(((i + alignW) >> 1) * blockC);
    }
  }

#if defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XE2)
  // Stores a row to a tensor using LSC 2D block stores which silently drop out-of-bounds writes
  template<typename SrcT, int N, typename DstT, TensorLayout layout>
  oidn_inline void storeRow(simd<SrcT, N>& row,
                            const TensorAccessor3D<DstT, layout>& dst,
                            int oc, int oh, int ow)
  {
    constexpr int blockC = TensorByteOffset<DstT, layout>::blockC;
    constexpr int lscBlockN = 64 / sizeof(DstT); // LSC 2D block width in elements
    static_assert(N % lscBlockN == 0, "row size must be multiple of LSC 2D block width");
    static_assert(TensorByteOffset<DstT, layout>::wByteStride % 32 == 0,
                  "Xe-HPC/Xe2+ 2D block surfaces need pixels of a multiple of 32 bytes");

    DstT* surfPtr = &dst(oc, 0, 0);
    const uint surfWidth  = dst.getByteOffset.hByteStride - 1; // bytes - 1
    const uint surfHeight = dst.H - 1;                         // rows - 1
    const uint surfPitch  = surfWidth;

    #pragma unroll
    for (int n = 0; n < N; n += lscBlockN)
    {
      store_2d<DstT, lscBlockN, 1>(
        surfPtr, surfWidth, surfHeight, surfPitch,
        ow * blockC + n, oh,
        row.template select<lscBlockN, 1>(n).read());
    }
  }

  // Stores multiple rows to a tensor using LSC 2D block stores which silently drop out-of-bounds writes
  template<int numRows, typename SrcT, int N, int M, typename DstT, TensorLayout layout>
  oidn_inline void storeRows(simd<SrcT, N> (&rows)[M],
                             const TensorAccessor3D<DstT, layout>& dst,
                             int oc, int oh, int ow)
  {
    constexpr int blockC = TensorByteOffset<DstT, layout>::blockC;
    constexpr int lscBlockN = 64 / sizeof(DstT); // LSC 2D block width in elements
    static_assert(N % lscBlockN == 0, "row size must be multiple of LSC 2D block width");
    static_assert(TensorByteOffset<DstT, layout>::wByteStride % 32 == 0,
                  "Xe-HPC/Xe2+ 2D block surfaces need pixels of a multiple of 32 bytes");

    DstT* surfPtr = &dst(oc, 0, 0);
    const uint surfWidth  = dst.getByteOffset.hByteStride - 1; // bytes - 1
    const uint surfHeight = dst.H - 1;                         // rows - 1
    const uint surfPitch  = surfWidth;

    #pragma unroll
    for (int n = 0; n < N; n += lscBlockN)
    {
      simd<DstT, lscBlockN * numRows> lscBlock;

      #pragma unroll
      for (int r = 0; r < numRows; ++r)
      {
        lscBlock.template select<lscBlockN, 1>(r * lscBlockN) =
          rows[r].template select<lscBlockN, 1>(n);
      }

      store_2d<DstT, lscBlockN, numRows>(
        surfPtr, surfWidth, surfHeight, surfPitch,
        ow * blockC + n, oh,
        lscBlock);
    }
  }
#else
  // Stores a row to a tensor using regular LSC block stores with explicit bounds checking
  template<typename SrcT, int N, typename DstT, TensorLayout layout>
  oidn_inline void storeRow(simd<SrcT, N>& row,
                            const TensorAccessor3D<DstT, layout>& dst,
                            int oc, int oh, int ow)
  {
    constexpr int blockC = TensorByteOffset<DstT, layout>::blockC;
    static_assert(N % blockC == 0, "non-integer width");
    constexpr int W = N / blockC;

    //if (oh >= dst.H)
    //  return;

    if (ow + W <= dst.W)
    {
      // Fast path: store the entire row
      DstT* dstPtr = &dst(oc, oh, ow);
      storeLargeBlock(dstPtr, row);
    }
    else
    {
      // Slow path: store the in-bounds pixels of the row
      const simd<int, W> owVec(ow, 1); // ow, ow+1, ow+2, ...
      simd_mask<W> predVec = owVec < dst.W;
      simd<uint32_t,  W> dstOffsetVec = dst.getByteOffset(oc, oh, 0) +
                                        owVec * dst.getByteOffset.wByteStride;
      simd<uintptr_t, W> dstAddrVec   = reinterpret_cast<uintptr_t>(dst.ptr) + dstOffsetVec;

      #pragma unroll
      for (int i = 0; i < W; ++i)
      {
        DstT* dstPtr = reinterpret_cast<DstT*>(uintptr_t(dstAddrVec[i]));
        storeBlock(dstPtr, row.template select<blockC, 1>(i * blockC).read(),
                   predVec.template select<1, 1>(i));
      }
    }
  }

  template<int numRows, typename SrcT, int N, int M, typename DstT, TensorLayout layout>
  oidn_inline void storeRows(simd<SrcT, N> (&rows)[M],
                             const TensorAccessor3D<DstT, layout>& dst,
                             int oc, int oh, int ow)
  {
    #pragma unroll
    for (int r = 0; r < numRows; ++r)
    {
      if (oh + r >= dst.H)
        break;
      storeRow(rows[r], dst, oc, oh + r, ow);
    }
  }
#endif

OIDN_NAMESPACE_END