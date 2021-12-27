// Copyright 2009-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "common/platform.h"

namespace oidn {

  // Tensor memory layout
  enum class TensorLayout
  {
    x,
    chw,
    Chw8c,  // blocked
    Chw16c, // blocked
    oihw,
  };

  template<TensorLayout layout>
  struct TensorLayoutTraits;

  template<>
  struct TensorLayoutTraits<TensorLayout::chw>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t wStride = sizeof(T);
      size_t hStride;
      size_t cStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int C, int H, int W)
      {
        hStride = size_t(W) * wStride;
        cStride = size_t(H) * hStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
      {
        return size_t(c) * cStride + size_t(h) * hStride + size_t(w) * wStride;
      }
    };
  };

  template<typename T, int blockSize>
  struct TensorAddressingChwBc
  {
    static constexpr int B = blockSize;

    static constexpr size_t wStride = B * sizeof(T);
    size_t hStride;
    size_t cStride;

    TensorAddressingChwBc() = default;

    OIDN_HOST_DEVICE_INLINE TensorAddressingChwBc(int C, int H, int W)
    {
      hStride = size_t(W) * wStride;
      cStride = size_t(H) * hStride;
    }

    OIDN_HOST_DEVICE_INLINE size_t getOffset(int c, int h, int w) const
    {
      return size_t(c/B) * cStride + size_t(h) * hStride + size_t(w) * wStride + size_t(c%B) * sizeof(T);
    }
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::Chw8c>
  {
    template<typename T>
    using Addressing = TensorAddressingChwBc<T, 8>;
  };
  
  template<>
  struct TensorLayoutTraits<TensorLayout::Chw16c>
  {
    template<typename T>
    using Addressing = TensorAddressingChwBc<T, 16>;
  };

  template<>
  struct TensorLayoutTraits<TensorLayout::oihw>
  {
    template<typename T>
    struct Addressing
    {
      static constexpr size_t wStride = sizeof(T);
      size_t hStride;
      size_t iStride;
      size_t oStride;

      Addressing() = default;

      OIDN_HOST_DEVICE_INLINE Addressing(int O, int I, int H, int W)
      {
        hStride = size_t(W) * wStride;
        iStride = size_t(H) * hStride;
        oStride = size_t(I) * iStride;
      }

      OIDN_HOST_DEVICE_INLINE size_t getOffset(int o, int i, int h, int w) const
      {
        return size_t(o) * oStride + size_t(i) * iStride + size_t(h) * hStride + size_t(w) * wStride;
      }
    };
  };

  template<typename T, TensorLayout layout>
  using TensorAddressing = typename TensorLayoutTraits<layout>::template Addressing<T>;

} // namespace oidn
