// Copyright 2009-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// FIXME: ESIMD compile error on Windows
#if defined(_WIN32)
  typedef unsigned int uint;
#endif

#include "common/platform.h"

namespace oidn {
  
  using namespace esimd;
  using namespace esimdx;

#if defined(OIDN_ARCH_XEHPC)
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

#if defined(OIDN_ARCH_XEHPC) || defined(OIDN_ARCH_XEHPG)

  template<typename T, int N>
  OIDN_INLINE simd<T, N> loadBlock(const T* ptr)
  {
    using DT = typename LSCBlockTraits<T, N>::DT;
    constexpr int DN = LSCBlockTraits<T, N>::DN;

    auto blk = lsc_block_load<DT, DN>((const DT*)ptr);
    return blk.template bit_cast_view<T>();
  }

  template<typename T, int N>
  OIDN_INLINE simd<T, N> loadBlock(const T* ptr, simd_mask<1> pred, simd<T, N> src = 0)
  {
    using DT = typename LSCBlockTraits<T, N>::DT;
    constexpr int DN = LSCBlockTraits<T, N>::DN;

    auto blk = lsc_block_load<DT, DN>((const DT*)ptr, pred);
    src.merge(blk.template bit_cast_view<T>(), simd_mask<N>(pred[0]));
    return src;
  }

  template<typename T, int N>
  OIDN_INLINE void storeBlock(T* ptr, simd<T, N> blk, simd_mask<1> pred = 1)
  {
    using DT = typename LSCBlockTraits<T, N>::DT;
    constexpr int DN = LSCBlockTraits<T, N>::DN;

    lsc_block_store<DT, DN>((DT*)ptr, blk.template bit_cast_view<DT>(), pred);
  }

#else

  template<typename T, int N>
  OIDN_INLINE simd<T, N> loadBlock(const T* ptr)
  {
    return block_load<T, N>(ptr, overaligned<16>);
  }

  template<typename T, int N>
  OIDN_INLINE simd<T, N> loadBlock(const T* ptr, simd_mask<1> pred, simd<T, N> src = 0)
  {
    if (pred)
      return block_load<T, N>(ptr, overaligned<16>);
    else
      return src;
  }

  template<typename T, int N>
  OIDN_INLINE void storeBlock(T* ptr, simd<T, N> blk)
  {
    block_store(ptr, blk);
  }

  template<typename T, int N>
  OIDN_INLINE void storeBlock(T* ptr, simd<T, N> blk, simd_mask<1> pred)
  {
    if (pred)
      block_store(ptr, blk);
  }

#endif

  template<typename T, int N, int blockSize = maxLSCBlockByteSize / sizeof(T), int offset = 0>
  OIDN_INLINE void loadLargeBlock(const T* ptr, simd<T, N>& dst)
  {
    if constexpr (offset + blockSize <= N)
    {
      dst.template select<blockSize, 1>(offset) = loadBlock<T, blockSize>(ptr + offset);
      loadLargeBlock<T, N, blockSize, offset + blockSize>(ptr, dst);
    }
    else if constexpr (offset < N)
      loadLargeBlock<T, N, blockSize / 2, offset>(ptr, dst);
  }

  template<typename T, int N, int blockSize = maxLSCBlockByteSize / sizeof(T), int offset = 0>
  OIDN_INLINE void storeLargeBlock(T* ptr, simd<T, N>& src)
  {
    if constexpr (offset + blockSize <= N)
    {
      storeBlock<T, blockSize>(ptr + offset, src.template select<blockSize, 1>(offset));
      storeLargeBlock<T, N, blockSize, offset + blockSize>(ptr, src);
    }
    else if constexpr (offset < N)
      storeLargeBlock<T, N, blockSize / 2, offset>(ptr, src);
  }

} // namespace oidn